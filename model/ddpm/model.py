import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from typing import Any, Optional, Tuple

class SinusoidalPosEmb(nn.Module):
    """
    Generates sinusoidal positional embeddings for diffusion timesteps.

    These embeddings are used to condition the UNet on the current diffusion step,
    allowing the model to behave differently at different stages of the process.

    Args:
        dim (int): The dimensionality of the output embeddings. Must be an even number.
    """
    def __init__(self, dim: int):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"Dimension for SinusoidalPosEmb must be even, got {dim}")
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        Args:
            time (torch.Tensor): A 1D tensor of timestep indices. Shape: (B,).

        Returns:
            torch.Tensor: Sinusoidal positional embeddings. Shape: (B, dim).
        """
        if time.ndim != 1:
            raise ValueError(f"Input time tensor must be 1D, got shape {time.shape}")
        
        device = time.device
        half_dim = self.dim // 2
        # Precompute the frequency term (div_term)
        # Formula: exp(arange(0, half_dim) * -(log(10000.0) / (half_dim - 1)))
        emb = np.log(10000) / (half_dim - 1 if half_dim > 1 else 1) # Avoid division by zero if half_dim is 1
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -emb)
        
        # emb shape: (half_dim,)
        # time shape: (B,) -> time[:, None] shape: (B, 1)
        # emb[None, :] shape: (1, half_dim)
        # Broadcasting results in (B, half_dim)
        emb = time[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1) # Concatenate sin and cos components
        return emb

class WeightStandardizedConv3d(nn.Conv3d):
    """
    Applies a 3D convolution with weight standardization.

    Weight standardization helps stabilize training by normalizing the weights
    of the convolutional kernel. It subtracts the mean and divides by the
    standard deviation of the weights.

    Inherits from nn.Conv3d, so all its arguments are accepted.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Tuple[int, int, int] = (3, 3, 3),
                 stride: Tuple[int, int, int] = (1, 1, 1), padding: Any = 1, 
                 dilation: Tuple[int, int, int] = (1, 1, 1), groups: int = 1, bias: bool = True):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, 
                         dilation=dilation, groups=groups, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor. Shape: (B, C_in, D_in, H_in, W_in).

        Returns:
            torch.Tensor: Output tensor. Shape: (B, C_out, D_out, H_out, W_out).
        """
        weight = self.weight
        # Standardize weights: (weight - mean) / (std + eps)
        # Calculate mean and std across output channels and spatial dimensions of the kernel
        weight_mean = weight.mean(dim=[1, 2, 3, 4], keepdim=True)
        weight_std = weight.std(dim=[1, 2, 3, 4], keepdim=True) + 1e-5 # Add epsilon for numerical stability
        weight = (weight - weight_mean) / weight_std
        
        return F.conv3d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class Downsample3D(nn.Module):
    """
    Downsamples a 3D tensor using a strided convolution.

    This module reduces the spatial dimensions (depth, height, width) typically by a factor of 2.

    Args:
        in_channels (int): Number of input channels.
        out_channels (Optional[int]): Number of output channels. If None, defaults to `in_channels`.
        kernel_size (int): Kernel size for the convolution. Default is 4.
        stride (int): Stride for the convolution. Default is 2.
        padding (int): Padding for the convolution. Default is 1.
    """
    def __init__(self, in_channels: int, out_channels: Optional[int] = None, 
                 kernel_size: int = 4, stride: int = 2, padding: int = 1):
        super().__init__()
        self.out_channels = out_channels if out_channels is not None else in_channels
        self.conv = nn.Conv3d(in_channels, self.out_channels, 
                              kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor. Shape: (B, C_in, D_in, H_in, W_in).

        Returns:
            torch.Tensor: Downsampled tensor. Shape: (B, C_out, D_out, H_out, W_out).
        """
        return self.conv(x)

class Upsample3D(nn.Module):
    """
    Upsamples a 3D tensor using nearest-neighbor interpolation followed by a convolution.

    This module increases the spatial dimensions (depth, height, width) typically by a factor of 2.

    Args:
        in_channels (int): Number of input channels.
        out_channels (Optional[int]): Number of output channels. If None, defaults to `in_channels`.
        scale_factor (int): Multiplier for spatial dimensions. Default is 2.
    """
    def __init__(self, in_channels: int, out_channels: Optional[int] = None, scale_factor: int = 2):
        super().__init__()
        self.out_channels = out_channels if out_channels is not None else in_channels
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='nearest')
        # Convolution after upsampling to refine features
        self.conv = nn.Conv3d(in_channels, self.out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor. Shape: (B, C_in, D_in, H_in, W_in).

        Returns:
            torch.Tensor: Upsampled tensor. Shape: (B, C_out, D_out, H_out, W_out).
        """
        x = self.upsample(x)
        x = self.conv(x)
        return x

class ResnetBlock3D(nn.Module):
    """
    A 3D residual block with time conditioning and group normalization.

    This block consists of two convolutional layers with normalization and activation,
    and a residual connection. Time embeddings can be incorporated to modulate
    the features.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        time_emb_dim (Optional[int]): Dimensionality of the time embeddings. If None,
                                      time conditioning is not applied.
        groups (int): Number of groups for GroupNormalization. Default is 8.
                      A common value is 32, ensure it divides `out_channels`.
    """
    def __init__(self, in_channels: int, out_channels: int, *, 
                 time_emb_dim: Optional[int] = None, groups: int = 8):
        super().__init__()

        if out_channels % groups != 0:
            # Fallback if groups don't divide out_channels, though ideally this should be configured correctly.
            # Find the largest factor of out_channels <= groups, or use 1 if no common factor.
            valid_groups = [g for g in range(1, groups + 1) if out_channels % g == 0]
            actual_groups = valid_groups[-1] if valid_groups else 1
            if actual_groups != groups:
                print(f"Warning: ResnetBlock3D groups changed from {groups} to {actual_groups} for out_channels {out_channels}")
            groups = actual_groups


        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels * 2) # Output scale and shift parameters
        ) if time_emb_dim is not None else None

        self.block1_conv = WeightStandardizedConv3d(in_channels, out_channels, kernel_size=(3,3,3), padding=1)
        self.norm1 = nn.GroupNorm(groups, out_channels)
        self.act1 = nn.SiLU() # Swish activation

        self.block2_conv = WeightStandardizedConv3d(out_channels, out_channels, kernel_size=(3,3,3), padding=1)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.act2 = nn.SiLU()

        # Residual connection: if in_channels != out_channels, use a 1x1 conv to match dimensions
        self.res_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor. Shape: (B, C_in, D, H, W).
            time_emb (Optional[torch.Tensor]): Time embeddings. Shape: (B, time_emb_dim).

        Returns:
            torch.Tensor: Output tensor. Shape: (B, C_out, D, H, W).
        """
        h = self.block1_conv(x)
        h = self.norm1(h)

        if self.mlp is not None and time_emb is not None:
            if time_emb.shape[0] != h.shape[0]:
                raise ValueError("Batch size mismatch between input tensor and time embeddings.")
            # time_emb: (B, time_emb_dim) -> mlp -> (B, out_channels * 2)
            time_encoding = self.mlp(time_emb)
            # Reshape for broadcasting: (B, out_channels * 2, 1, 1, 1)
            time_encoding = time_encoding.view(h.shape[0], -1, 1, 1, 1)
            # Split into scale and shift
            scale, shift = time_encoding.chunk(2, dim=1) # Each (B, out_channels, 1, 1, 1)
            h = h * (scale + 1) + shift # Apply scale and shift
        
        h = self.act1(h)
        h = self.block2_conv(h)
        h = self.norm2(h)
        h = self.act2(h)

        return h + self.res_conv(x) # Add residual connection


class Attention3D(nn.Module):
    """
    Standard 3D multi-head self-attention mechanism.

    Computes attention over the 3D spatial dimensions (depth, height, width).
    Uses Group Normalization before attention.

    Args:
        in_channels (int): Number of input channels.
        heads (int): Number of attention heads. Default is 4.
        dim_head (int): Dimensionality of each attention head. Default is 32.
        groups_for_norm (int): Number of groups for the GroupNorm layer. Default is 32.
                               Ensure this divides `in_channels`.
    """
    def __init__(self, in_channels: int, heads: int = 4, dim_head: int = 32, groups_for_norm: int = 32):
        super().__init__()
        self.scale = dim_head ** -0.5 # For scaled dot-product attention
        self.heads = heads
        hidden_dim = dim_head * heads # Total dimension for Q, K, V

        if in_channels % groups_for_norm != 0:
            valid_groups = [g for g in range(1, groups_for_norm + 1) if in_channels % g == 0]
            actual_groups = valid_groups[-1] if valid_groups else 1
            if actual_groups != groups_for_norm:
                 print(f"Warning: Attention3D groups_for_norm changed from {groups_for_norm} to {actual_groups} for in_channels {in_channels}")
            groups_for_norm = actual_groups
        
        self.norm = nn.GroupNorm(groups_for_norm, in_channels)
        self.to_qkv = nn.Conv3d(in_channels, hidden_dim * 3, kernel_size=1, bias=False) # Project to Q, K, V
        self.to_out = nn.Conv3d(hidden_dim, in_channels, kernel_size=1) # Project back to original channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor. Shape: (B, C, D, H, W).

        Returns:
            torch.Tensor: Output tensor after attention. Shape: (B, C, D, H, W).
        """
        B, C, D, H, W = x.shape
        
        residual = x # Store for residual connection
        x_norm = self.norm(x)
        
        # qkv shape: (B, hidden_dim * 3, D, H, W)
        qkv = self.to_qkv(x_norm).chunk(3, dim=1) # Split into Q, K, V (each B, hidden_dim, D, H, W)
        
        # Rearrange for multi-head attention: (B, heads, D*H*W, dim_head)
        q, k, v = map(
            lambda t: rearrange(t, 'b (heads c_h) d h w -> b heads (d h w) c_h', heads=self.heads), 
            qkv
        )
        
        # Scaled dot-product attention: (Q K^T) / sqrt(d_k)
        # q: (B, heads, N, dim_head), k: (B, heads, N, dim_head) -> k.transpose: (B, heads, dim_head, N)
        # dots: (B, heads, N, N) where N = D*H*W
        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k.transpose(-2, -1)) * self.scale
        attn_weights = dots.softmax(dim=-1) # Apply softmax over the last dimension (keys)
        
        # Apply attention weights to values: (B, heads, N, dim_head)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn_weights, v)
        
        # Rearrange back to original spatial format: (B, hidden_dim, D, H, W)
        out = rearrange(out, 'b heads (d h w) c_h -> b (heads c_h) d h w', heads=self.heads, d=D, h=H, w=W)
        out = self.to_out(out) # Project back
        
        return out + residual # Add residual connection


class LinearAttention3D(nn.Module):
    """
    Linear 3D multi-head attention mechanism (memory-efficient alternative).

    Applies softmax to queries and keys separately before their dot product,
    reducing memory complexity from O(N^2) to O(N).

    Args:
        in_channels (int): Number of input channels.
        heads (int): Number of attention heads. Default is 4.
        dim_head (int): Dimensionality of each attention head. Default is 32.
        groups_for_norm (int): Number of groups for GroupNorm layers. Default is 32.
    """
    def __init__(self, in_channels: int, heads: int = 4, dim_head: int = 32, groups_for_norm: int = 32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads

        if in_channels % groups_for_norm != 0:
            valid_groups = [g for g in range(1, groups_for_norm + 1) if in_channels % g == 0]
            actual_groups = valid_groups[-1] if valid_groups else 1
            if actual_groups != groups_for_norm:
                 print(f"Warning: LinearAttention3D groups_for_norm changed from {groups_for_norm} to {actual_groups} for in_channels {in_channels}")
            groups_for_norm = actual_groups

        self.norm = nn.GroupNorm(groups_for_norm, in_channels)
        self.to_qkv = nn.Conv3d(in_channels, hidden_dim * 3, kernel_size=1, bias=False)
        
        self.to_out = nn.Sequential(
            nn.Conv3d(hidden_dim, in_channels, kernel_size=1),
            nn.GroupNorm(groups_for_norm, in_channels) # Normalize output
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor. Shape: (B, C, D, H, W).

        Returns:
            torch.Tensor: Output tensor after linear attention. Shape: (B, C, D, H, W).
        """
        B, C, D_spatial, H_spatial, W_spatial = x.shape # Use different names for spatial dims to avoid conflict
        residual = x
        x_norm = self.norm(x)

        qkv = self.to_qkv(x_norm).chunk(3, dim=1)
        # Rearrange q, k, v to (B, heads, dim_head, N) where N = D*H*W
        q, k, v = map(
            lambda t: rearrange(t, 'b (h d_h) d h w -> b h d_h (d h w)', h=self.heads), 
            qkv
        )

        # Apply softmax to queries (dim=-1, features) and keys (dim=-2, sequence length)
        # In linear attention, softmax is often applied along different axes.
        # Original "Attention is All You Need" applies softmax to QK^T.
        # Linear attention variants (e.g., Linformer, Performer) use kernel methods or other tricks.
        # A common simplification for "linear attention" is softmax on Q and K independently.
        q_softmax = q.softmax(dim=-1) # Softmax over features for Q
        k_softmax = k.softmax(dim=-2) # Softmax over spatial dimension for K
                                      # (or dim=-1 if thinking of it as features of K)

        # Optional scaling, sometimes seen in linear attention variants
        # q_softmax = q_softmax / (v.shape[-1] ** 0.25) # Example scaling
        # k_softmax = k_softmax / (v.shape[-1] ** 0.25)

        # Compute context: K^T V (or similar, depending on the linear attention variant)
        # k_softmax: (B, heads, dim_head, N) -> k_softmax.transpose: (B, heads, N, dim_head)
        # v: (B, heads, dim_head, N)
        # context: (B, heads, dim_head, dim_head) if K^T V
        # Here, if we do k_softmax @ v.transpose(-1, -2) -> (B, h, d_h, N) @ (B, h, N, d_h) -> (B, h, d_h, d_h)
        # Or, if we want to sum over N for K:
        context = torch.einsum('b h d n, b h e n -> b h d e', k_softmax, v) # K V^T (sum over N) -> (B, h, d_h, d_h_v)
                                                                        # Here d_h_v is dim_head of V

        # Apply Q to context
        # out: (B, heads, dim_head, N)
        out = torch.einsum('b h d e, b h d n -> b h e n', context, q_softmax) # Q (K V^T)
        
        # Rearrange back: (B, hidden_dim, D, H, W)
        out = rearrange(out, 'b (h e) d h w -> b (h e) d h w', 
                        h=self.heads, d=D_spatial, h=H_spatial, w=W_spatial, e=self.to_qkv.out_channels // (3 * self.heads)) # Correct 'e'
        out = self.to_out(out)
        
        return out + residual

class AttnBlock3D(nn.Module):
    """
    A 3D attention block that wraps either standard or linear attention.

    Args:
        in_channels (int): Number of input channels.
        heads (int): Number of attention heads. Default is 4.
        dim_head (int): Dimensionality of each attention head. Default is 32.
        use_linear_attention (bool): If True, uses LinearAttention3D; otherwise, Attention3D.
                                     Default is True.
        groups_for_norm (int): Number of groups for GroupNorm in the attention layers.
    """
    def __init__(self, in_channels: int, heads: int = 4, dim_head: int = 32, 
                 use_linear_attention: bool = True, groups_for_norm: int = 32):
        super().__init__()
        if use_linear_attention:
            self.attn = LinearAttention3D(in_channels, heads, dim_head, groups_for_norm=groups_for_norm)
        else:
            self.attn = Attention3D(in_channels, heads, dim_head, groups_for_norm=groups_for_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor. Shape: (B, C, D, H, W).

        Returns:
            torch.Tensor: Output tensor after attention. Shape: (B, C, D, H, W).
        """
        return self.attn(x)


class UNet3D(nn.Module):
    """
    A 3D U-Net architecture for diffusion models.

    This network takes a noisy 3D voxel grid and a timestep embedding as input,
    and typically predicts either the noise added to the grid or the clean grid itself.
    It features a downsampling path, a bottleneck, and an upsampling path with
    skip connections, incorporating ResNet blocks and attention mechanisms.

    Args:
        dim (int): Base channel dimension for the U-Net.
        dim_mults (Tuple[int, ...]): Multipliers for channel dimensions at each
                                     resolution level in the downsampling/upsampling paths.
                                     Example: (1, 2, 4, 8)
        in_channels (int): Number of input channels of the voxel data (e.g., embedding dimension).
                           If self-conditioning is used, this should be `embedding_dim * 2`.
        out_dim (Optional[int]): Number of output channels. If None, defaults to `in_channels`
                                 (or `in_channels // 2` if self-conditioning doubled input).
        init_dim (Optional[int]): Channel dimension after the initial convolution.
                                  If None, defaults to `dim`.
        resnet_block_groups (int): Number of groups for GroupNormalization in ResnetBlock3D.
        use_linear_attention (bool): Whether to use LinearAttention3D instead of Attention3D.
        attn_heads (int): Number of heads for attention mechanisms.
        attn_dim_head (int): Dimension per head for attention mechanisms.
    """
    def __init__(self,
                 dim: int,
                 dim_mults: Tuple[int, ...],
                 in_channels: int,
                 out_dim: Optional[int] = None,
                 init_dim: Optional[int] = None,
                 resnet_block_groups: int = 8,
                 use_linear_attention: bool = True,
                 attn_heads: int = 4,
                 attn_dim_head: int = 32):
        super().__init__()

        self.in_channels = in_channels
        self.init_dim = init_dim if init_dim is not None else dim
        
        # Determine output dimension. If self-conditioning doubles input channels,
        # the typical output is the original embedding dimension.
        if out_dim is None:
            # Heuristic: if in_channels seems doubled (e.g. for self-cond), out_dim is half.
            # This needs to be set carefully by the user based on `PRED_X0` and `SELF_CONDITION` flags.
            # For now, let's assume out_dim is explicitly passed or defaults to a sensible value.
            # A common case: if model predicts noise/x0, out_dim = original_embedding_dim
            self.out_dim = in_channels 
        else:
            self.out_dim = out_dim


        # Time embedding projection
        time_mlp_dim = dim * 4 # Standard dimension for projected time embeddings
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim), # Input to SinusoidalPosEmb is `dim`
            nn.Linear(dim, time_mlp_dim),
            nn.GELU(),
            nn.Linear(time_mlp_dim, time_mlp_dim)
        )

        # Initial convolution: projects input voxel data to `init_dim` channels
        self.init_conv = nn.Conv3d(self.in_channels, self.init_dim, kernel_size=7, padding=3)

        # Calculate channel dimensions for each resolution level
        # Example: if dim=64, dim_mults=(1,2,4), then dims = [init_dim, 64, 128, 256]
        dims = [self.init_dim, *map(lambda m: dim * m, dim_mults)]
        # Create pairs of (in_channels, out_channels) for each down/up block
        in_out = list(zip(dims[:-1], dims[1:]))
        num_resolutions = len(in_out)

        # --- Downsampling Path ---
        self.downs = nn.ModuleList([])
        for i, (dim_in, dim_out) in enumerate(in_out):
            is_last_resolution = (i >= (num_resolutions - 1))
            self.downs.append(nn.ModuleList([
                ResnetBlock3D(dim_in, dim_out, time_emb_dim=time_mlp_dim, groups=resnet_block_groups),
                ResnetBlock3D(dim_out, dim_out, time_emb_dim=time_mlp_dim, groups=resnet_block_groups),
                AttnBlock3D(dim_out, heads=attn_heads, dim_head=attn_dim_head, 
                            use_linear_attention=use_linear_attention, groups_for_norm=resnet_block_groups),
                Downsample3D(dim_out, dim_out) if not is_last_resolution else nn.Identity()
            ]))

        # --- Bottleneck ---
        mid_dim = dims[-1] # Dimension at the bottleneck
        self.mid_block1 = ResnetBlock3D(mid_dim, mid_dim, time_emb_dim=time_mlp_dim, groups=resnet_block_groups)
        self.mid_attn = AttnBlock3D(mid_dim, heads=attn_heads, dim_head=attn_dim_head, 
                                    use_linear_attention=use_linear_attention, groups_for_norm=resnet_block_groups)
        self.mid_block2 = ResnetBlock3D(mid_dim, mid_dim, time_emb_dim=time_mlp_dim, groups=resnet_block_groups)

        # --- Upsampling Path ---
        self.ups = nn.ModuleList([])
        # Iterate in reverse, excluding the initial `init_dim` to `dims[1]` transition
        for i, (dim_in, dim_out) in enumerate(reversed(in_out[1:])): 
            is_last_resolution = (i >= (num_resolutions - 2)) # Adjusted for reversed iteration and skipping one
            self.ups.append(nn.ModuleList([
                # Input to ResNet block is dim_out (from previous upsample) + dim_in (from skip connection)
                ResnetBlock3D(dim_out + dim_in, dim_in, time_emb_dim=time_mlp_dim, groups=resnet_block_groups),
                ResnetBlock3D(dim_in, dim_in, time_emb_dim=time_mlp_dim, groups=resnet_block_groups),
                AttnBlock3D(dim_in, heads=attn_heads, dim_head=attn_dim_head, 
                            use_linear_attention=use_linear_attention, groups_for_norm=resnet_block_groups),
                Upsample3D(dim_in, dim_in) if not is_last_resolution else nn.Identity()
            ]))
        
        # Final convolution: projects features to the desired output dimension
        # The input to this block will be `dim` (from the last upsampling stage)
        # concatenated with the skip connection from the `init_conv` stage (which has `init_dim` channels).
        # If init_dim == dim, then input is dim * 2.
        final_conv_in_channels = dim + self.init_dim # Corrected based on skip connection from init_conv
        if self.init_dim != dim: # A common setup is init_dim = dim
             print(f"UNet3D Warning: init_dim ({self.init_dim}) != dim ({dim}). Final conv input channels: {final_conv_in_channels}")

        self.final_conv = nn.Sequential(
            ResnetBlock3D(final_conv_in_channels, dim, time_emb_dim=time_mlp_dim, groups=resnet_block_groups),
            nn.Conv3d(dim, self.out_dim, kernel_size=1) # Final projection to out_dim
        )

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input noisy voxel tensor.
                              Shape: (B, C_in, D, H, W). C_in is `self.in_channels`.
            time (torch.Tensor): Timestep indices. Shape: (B,).

        Returns:
            torch.Tensor: Output tensor (predicted noise or x0).
                          Shape: (B, C_out, D, H, W). C_out is `self.out_dim`.
        """
        # 1. Time embedding
        t_emb = self.time_mlp(time) # (B, time_mlp_dim)
        
        # 2. Initial convolution
        x = self.init_conv(x) # (B, self.init_dim, D, H, W)
        
        # Store for skip connection to the final layer
        h_init = x.clone() 
        
        skip_connections = [] # For skip connections from downsampling path

        # 3. Downsampling Path
        for resnet_block1, resnet_block2, attn, downsample in self.downs:
            x = resnet_block1(x, t_emb)
            x = resnet_block2(x, t_emb)
            x = attn(x)
            skip_connections.append(x) # Store for upsampling path
            x = downsample(x)
            
        # 4. Bottleneck
        x = self.mid_block1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t_emb)

        # 5. Upsampling Path
        for resnet_block1, resnet_block2, attn, upsample in self.ups:
            skip = skip_connections.pop()
            x = torch.cat((x, skip), dim=1) # Concatenate skip connection
            
            x = resnet_block1(x, t_emb)
            x = resnet_block2(x, t_emb)
            x = attn(x)
            x = upsample(x)
            
        # 6. Final Layer
        # Concatenate with the skip connection from the initial convolution output
        x = torch.cat((x, h_init), dim=1) # CRITICAL FIX: Added this skip connection
        
        x = self.final_conv(x) # (B, self.out_dim, D, H, W)
        return x
