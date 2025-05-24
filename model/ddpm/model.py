import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from typing import Any, Optional, Tuple
import logging

if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO)

class SinusoidalPosEmb(nn.Module):
    """
    Generates sinusoidal positional embeddings for diffusion timesteps.
    Args:
        dim (int): The dimensionality of the output embeddings. Must be an even number.
    """
    def __init__(self, dim: int):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"Dimension for SinusoidalPosEmb must be even, got {dim}")
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        if time.ndim != 1:
            raise ValueError(f"Input time tensor must be 1D, got shape {time.shape}")
        device = time.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1 if half_dim > 1 else 1) 
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -emb)
        emb = time[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class WeightStandardizedConv3d(nn.Conv3d):
    """Applies a 3D convolution with weight standardization."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Tuple[int, int, int] = (3, 3, 3),
                 stride: Tuple[int, int, int] = (1, 1, 1), padding: Any = 1, 
                 dilation: Tuple[int, int, int] = (1, 1, 1), groups: int = 1, bias: bool = True):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, 
                         dilation=dilation, groups=groups, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        weight_mean = weight.mean(dim=[1, 2, 3, 4], keepdim=True)
        weight_std = weight.std(dim=[1, 2, 3, 4], keepdim=True) + 1e-5
        weight = (weight - weight_mean) / weight_std
        return F.conv3d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class Downsample3D(nn.Module):
    """Downsamples a 3D tensor using a strided convolution."""
    def __init__(self, in_channels: int, out_channels: Optional[int] = None, 
                 kernel_size: int = 4, stride: int = 2, padding: int = 1):
        super().__init__()
        self.out_channels = out_channels if out_channels is not None else in_channels
        self.conv = nn.Conv3d(in_channels, self.out_channels, 
                              kernel_size=kernel_size, stride=stride, padding=padding)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class Upsample3D(nn.Module):
    """Upsamples a 3D tensor using nearest-neighbor interpolation followed by a convolution."""
    def __init__(self, in_channels: int, out_channels: Optional[int] = None, scale_factor: int = 2):
        super().__init__()
        self.out_channels = out_channels if out_channels is not None else in_channels
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='nearest')
        self.conv = nn.Conv3d(in_channels, self.out_channels, kernel_size=3, padding=1) # Conv to refine after upsample
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = self.conv(x)
        return x

class ResnetBlock3D(nn.Module):
    """A 3D residual block with time conditioning and group normalization."""
    def __init__(self, in_channels: int, out_channels: int, *, 
                 time_emb_dim: Optional[int] = None, groups: int = 8):
        super().__init__()
        self.in_channels = in_channels 
        self.out_channels = out_channels 

        if out_channels > 0 and out_channels % groups != 0 :
            valid_groups = [g for g in range(1, groups + 1) if out_channels % g == 0]
            actual_groups = valid_groups[-1] if valid_groups else 1
            if actual_groups != groups:
                logging.warning(f"[ResnetBlock3D] groups changed from {groups} to {actual_groups} for out_channels {out_channels}")
            groups = actual_groups
        elif out_channels == 0: 
            groups = 1

        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_channels * 2)) if time_emb_dim is not None else None
        self.block1_conv = WeightStandardizedConv3d(in_channels, out_channels, kernel_size=(3,3,3), padding=1)
        self.norm1 = nn.GroupNorm(groups, out_channels) if out_channels > 0 else nn.Identity()
        self.act1 = nn.SiLU()
        self.block2_conv = WeightStandardizedConv3d(out_channels, out_channels, kernel_size=(3,3,3), padding=1)
        self.norm2 = nn.GroupNorm(groups, out_channels) if out_channels > 0 else nn.Identity()
        self.act2 = nn.SiLU()
        self.res_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.block1_conv(x)
        h = self.norm1(h)
        if self.mlp is not None and time_emb is not None and h.shape[1] > 0:
            if time_emb.shape[0] != h.shape[0]: raise ValueError("Batch size mismatch between input and time_emb.")
            time_encoding = self.mlp(time_emb)
            time_encoding = time_encoding.view(h.shape[0], -1, 1, 1, 1)
            scale, shift = time_encoding.chunk(2, dim=1)
            h = h * (scale + 1) + shift
        h = self.act1(h)
        h = self.block2_conv(h)
        h = self.norm2(h)
        h = self.act2(h)
        res_out = self.res_conv(x)
        return h + res_out

class Attention3D(nn.Module):
    """Standard 3D multi-head self-attention mechanism."""
    def __init__(self, in_channels: int, heads: int = 4, dim_head: int = 32, groups_for_norm: int = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        if in_channels > 0 and in_channels % groups_for_norm != 0:
            valid_groups = [g for g in range(1, groups_for_norm + 1) if in_channels % g == 0]; actual_groups = valid_groups[-1] if valid_groups else 1
            if actual_groups != groups_for_norm: logging.warning(f"Attention3D groups_for_norm changed from {groups_for_norm} to {actual_groups} for in_channels {in_channels}")
            groups_for_norm = actual_groups
        elif in_channels == 0: groups_for_norm = 1
        self.norm = nn.GroupNorm(groups_for_norm, in_channels) if in_channels > 0 else nn.Identity()
        self.to_qkv = nn.Conv3d(in_channels, hidden_dim * 3, kernel_size=1, bias=False)
        self.to_out = nn.Conv3d(hidden_dim, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        if C == 0: return x 
        residual = x; x_norm = self.norm(x); qkv = self.to_qkv(x_norm).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (heads c_h) d h w -> b heads (d h w) c_h', heads=self.heads), qkv)
        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k.transpose(-2, -1)) * self.scale
        attn_weights = dots.softmax(dim=-1)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn_weights, v)
        out = rearrange(out, 'b heads (d h w) c_h -> b (heads c_h) d h w', heads=self.heads, d=D, h=H, w=W)
        out = self.to_out(out)
        return out + residual

class LinearAttention3D(nn.Module):
    """Linear 3D multi-head attention mechanism."""
    def __init__(self, in_channels: int, heads: int = 4, dim_head: int = 32, groups_for_norm: int = 32):
        super().__init__()
        self.heads = heads; self.dim_head = dim_head; hidden_dim = dim_head * heads
        if in_channels > 0 and in_channels % groups_for_norm != 0:
            valid_groups = [g for g in range(1, groups_for_norm + 1) if in_channels % g == 0]; actual_groups = valid_groups[-1] if valid_groups else 1
            if actual_groups != groups_for_norm: logging.warning(f"LinearAttention3D groups_for_norm changed from {groups_for_norm} to {actual_groups} for in_channels {in_channels}")
            groups_for_norm = actual_groups
        elif in_channels == 0: groups_for_norm = 1
        self.norm = nn.GroupNorm(groups_for_norm, in_channels) if in_channels > 0 else nn.Identity()
        self.to_qkv = nn.Conv3d(in_channels, hidden_dim * 3, kernel_size=1, bias=False)
        self.to_out = nn.Sequential(nn.Conv3d(hidden_dim, in_channels, kernel_size=1), nn.GroupNorm(groups_for_norm, in_channels) if in_channels > 0 else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D_spatial, H_spatial, W_spatial = x.shape
        if C == 0: return x
        residual = x; x_norm = self.norm(x); qkv = self.to_qkv(x_norm).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h dh) dsp hsp wsp -> b h dh (dsp hsp wsp)', h=self.heads), qkv)
        q_softmax = q.softmax(dim=-1); k_softmax = k.softmax(dim=-2)
        context = torch.einsum('b h dn, b h en -> b h de', k_softmax, v) 
        out_intermediate = torch.einsum('b h de, b h dn -> b h en', context, q_softmax)
        out = rearrange(out_intermediate, 'b num_h dim_h (sp_d sp_h sp_w) -> b (num_h dim_h) sp_d sp_h sp_w', sp_d=D_spatial, sp_h=H_spatial, sp_w=W_spatial)
        out = self.to_out(out)
        return out + residual

class AttnBlock3D(nn.Module):
    """A 3D attention block."""
    def __init__(self, in_channels: int, heads: int = 4, dim_head: int = 32, use_linear_attention: bool = True, groups_for_norm: int = 32):
        super().__init__()
        if use_linear_attention: self.attn = LinearAttention3D(in_channels, heads, dim_head, groups_for_norm=groups_for_norm)
        else: self.attn = Attention3D(in_channels, heads, dim_head, groups_for_norm=groups_for_norm)
    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.attn(x)

class UNet3D(nn.Module):
    """A 3D U-Net architecture for diffusion models."""
    def __init__(self, dim: int, dim_mults: Tuple[int, ...], in_channels: int, out_dim: Optional[int] = None,
                 init_dim: Optional[int] = None, resnet_block_groups: int = 8, use_linear_attention: bool = True,
                 attn_heads: int = 4, attn_dim_head: int = 32):
        super().__init__()
        self.in_channels = in_channels; self.init_dim = init_dim if init_dim is not None else dim
        self.out_dim = out_dim if out_dim is not None else self.in_channels
        if out_dim is None: logging.warning(f"[UNet3D Init] out_dim not specified, defaulting to in_channels ({self.in_channels}).")

        time_mlp_dim = dim * 4
        self.time_mlp = nn.Sequential(SinusoidalPosEmb(dim), nn.Linear(dim, time_mlp_dim), nn.GELU(), nn.Linear(time_mlp_dim, time_mlp_dim))
        self.init_conv = nn.Conv3d(self.in_channels, self.init_dim, kernel_size=7, padding=3)
        logging.debug(f"[UNet3D Init] init_conv: in_channels={self.in_channels}, out_channels={self.init_dim}")

        dims = [self.init_dim, *map(lambda m: dim * m, dim_mults)]
        logging.debug(f"[UNet3D Init] Channel dimensions per level (after init_conv then mults): {dims}")
        in_out = list(zip(dims[:-1], dims[1:]))
        self.num_resolutions = len(in_out)
        logging.debug(f"[UNet3D Init] Num resolutions (down/up stages): {self.num_resolutions}")

        self.downs = nn.ModuleList([])
        for i, (d_in, d_out) in enumerate(in_out):
            is_last_res_in_downs = (i >= (self.num_resolutions - 1))
            logging.debug(f"[UNet3D Init] Down Block {i}: ResNet({d_in}->{d_out}), ResNet({d_out}->{d_out}), Attn({d_out}), Downsample={not is_last_res_in_downs}")
            self.downs.append(nn.ModuleList([
                ResnetBlock3D(d_in, d_out, time_emb_dim=time_mlp_dim, groups=resnet_block_groups),
                ResnetBlock3D(d_out, d_out, time_emb_dim=time_mlp_dim, groups=resnet_block_groups),
                AttnBlock3D(d_out, heads=attn_heads, dim_head=attn_dim_head, use_linear_attention=use_linear_attention, groups_for_norm=resnet_block_groups),
                Downsample3D(d_out, d_out) if not is_last_res_in_downs else nn.Identity()]))

        mid_dim = dims[-1]
        logging.debug(f"[UNet3D Init] Bottleneck mid_dim: {mid_dim}")
        self.mid_block1 = ResnetBlock3D(mid_dim, mid_dim, time_emb_dim=time_mlp_dim, groups=resnet_block_groups)
        self.mid_attn = AttnBlock3D(mid_dim, heads=attn_heads, dim_head=attn_dim_head, use_linear_attention=use_linear_attention, groups_for_norm=resnet_block_groups)
        self.mid_block2 = ResnetBlock3D(mid_dim, mid_dim, time_emb_dim=time_mlp_dim, groups=resnet_block_groups)

        self.ups = nn.ModuleList([])
        for i, (ch_skip_target, ch_deep_in) in enumerate(reversed(in_out[1:])):
            target_out_channels_for_stage = ch_skip_target
            logging.debug(f"[UNet3D Init] Up Block {i}: Upsample({ch_deep_in}->{ch_deep_in}), "
                          f"Concat_Input_ResNet1({ch_deep_in} + {ch_skip_target} -> {target_out_channels_for_stage}), "
                          f"ResNet2({target_out_channels_for_stage}->{target_out_channels_for_stage}), Attn({target_out_channels_for_stage})")
            self.ups.append(nn.ModuleList([
                Upsample3D(ch_deep_in, ch_deep_in),
                ResnetBlock3D(ch_deep_in + ch_skip_target, target_out_channels_for_stage, time_emb_dim=time_mlp_dim, groups=resnet_block_groups),
                ResnetBlock3D(target_out_channels_for_stage, target_out_channels_for_stage, time_emb_dim=time_mlp_dim, groups=resnet_block_groups),
                AttnBlock3D(target_out_channels_for_stage, heads=attn_heads, dim_head=attn_dim_head, use_linear_attention=use_linear_attention, groups_for_norm=resnet_block_groups)
            ]))
        
        final_conv_in_channels = dims[1] + self.init_dim
        logging.debug(f"[UNet3D Init] Final conv input_channels: {final_conv_in_channels} (output_of_ups={dims[1]} + init_dim={self.init_dim}), output_channels_before_proj={dim}, final_out_dim={self.out_dim}")
        self.final_conv = nn.Sequential(
            ResnetBlock3D(final_conv_in_channels, dim, time_emb_dim=time_mlp_dim, groups=resnet_block_groups),
            nn.Conv3d(dim, self.out_dim, kernel_size=1))

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        logging.debug(f"[UNet3D Fwd] Initial x shape: {x.shape}, expected in_channels: {self.in_channels}")
        if x.shape[1] != self.in_channels:
            logging.error(f"[UNet3D Fwd] FATAL: Input channel mismatch! Expected {self.in_channels}, got {x.shape[1]}")
            # Consider raising an error here if this check fails.
            # For now, proceeding might lead to more errors but shows the flow.

        t_emb = self.time_mlp(time)
        x = self.init_conv(x)
        logging.debug(f"[UNet3D Fwd] After init_conv, x shape: {x.shape} (init_dim={self.init_dim})")
        h_init = x.clone() 
        
        skip_connections = []
        for i_down, (resnet_block1, resnet_block2, attn, downsample_module) in enumerate(self.downs):
            # logging.debug(f"  [UNet3D Fwd] Down Block {i_down} - Input x shape: {x.shape}")
            x = resnet_block1(x, t_emb)
            x = resnet_block2(x, t_emb)
            x = attn(x)
            # logging.debug(f"  [UNet3D Fwd] Down Block {i_down} - Output x shape before downsample: {x.shape}")
            if i_down < self.num_resolutions - 1: 
                skip_connections.append(x)
                # logging.debug(f"  [UNet3D Fwd] Down Block {i_down} - Stored skip shape: {x.shape}")
            x = downsample_module(x) 
            
        # logging.debug(f"[UNet3D Fwd] After Down Blocks, x shape (to bottleneck): {x.shape}")
        x = self.mid_block1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t_emb) 
        logging.debug(f"[UNet3D Fwd] After Bottleneck, x shape: {x.shape}")

        for i_up, (upsample_module, resnet_block1, resnet_block2, attn) in enumerate(self.ups):
            x = upsample_module(x) 
            logging.debug(f"[UNet3D Fwd] Up Block {i_up} - x shape AFTER upsample: {x.shape}")
            
            if not skip_connections:
                logging.error(f"[UNet3D Fwd] FATAL: Ran out of skip connections for Up Block {i_up}!")
                raise ValueError("Mismatch in skip connections and upsampling stages.")
            skip = skip_connections.pop()
            logging.debug(f"                 skip shape for Up Block {i_up}: {skip.shape}")
            
            if x.shape[0] != skip.shape[0] or x.shape[2:] != skip.shape[2:]: 
                 logging.error(f"[UNet3D Fwd] FATAL: Spatial or batch dim mismatch for skip connection in Up Block {i_up} AFTER upsample!")
                 logging.error(f"                x: {x.shape}, skip: {skip.shape}")
                 raise ValueError("Spatial/batch dim mismatch for skip connection AFTER upsample.")

            x = torch.cat((x, skip), dim=1)
            logging.debug(f"[UNet3D Fwd] Up Block {i_up} - x shape after cat (input to ResNet1): {x.shape}")
            
            x = resnet_block1(x, t_emb)
            x = resnet_block2(x, t_emb)
            x = attn(x)
            
        x = torch.cat((x, h_init), dim=1) 
        logging.debug(f"[UNet3D Fwd] After final cat, x shape (input to final_conv ResNet): {x.shape}")
        
        x = self.final_conv(x)
        logging.debug(f"[UNet3D Fwd] Final output x shape: {x.shape}")
        return x
