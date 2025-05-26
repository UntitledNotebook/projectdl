# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Time Embedding Modules ---

class SinusoidalTimeEmbedding(nn.Module):
    """
    Generates sinusoidal Fourier features for continuous time t.
    """
    def __init__(self, embedding_dim: int, max_period: float = 10000.0):
        super().__init__()
        if embedding_dim % 2 != 0:
            raise ValueError(f"embedding_dim must be even, got {embedding_dim}")
        self.embedding_dim = embedding_dim
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        original_shape = t.shape
        if t.ndim > 1:
            t = t.squeeze()
        if t.ndim == 0: 
            t = t.unsqueeze(0)
        assert t.ndim == 1, f"Input time tensor t must be effectively 1D (batch_size,), got shape {original_shape} -> {t.shape} after squeeze"
        
        half_dim = self.embedding_dim // 2
        indices = torch.arange(half_dim, device=t.device, dtype=torch.float32)
        omegas = torch.exp(indices * (-math.log(self.max_period) / half_dim))
        args = t.unsqueeze(1) * omegas.unsqueeze(0)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return embedding

class FullTimeEmbedding(nn.Module):
    """
    Combines sinusoidal features with an MLP for a final time embedding.
    """
    def __init__(self, 
                 fourier_embedding_dim: int,
                 mlp_hidden_dim: int,
                 final_embedding_dim: int,
                 max_period: float = 10000.0):
        super().__init__()
        self.fourier_features = SinusoidalTimeEmbedding(fourier_embedding_dim, max_period)
        self.mlp = nn.Sequential(nn.Linear(fourier_embedding_dim, mlp_hidden_dim),
                                  nn.SiLU(),
                                  nn.Linear(mlp_hidden_dim, final_embedding_dim))
        self.final_embedding_dim = final_embedding_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.fourier_features(t))

# --- Convolution Layer ---
class WSConv3d(nn.Conv3d):
    """
    Weighted Standardized 3D Convolution.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros',
                 eps=1e-5):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias, padding_mode)
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(self.out_channels, 1, 1, 1))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        mean = weight.mean(dim=[1, 2, 3, 4], keepdim=True)
        std = weight.std(dim=[1, 2, 3, 4], keepdim=True) + self.eps 
        standardized_weight = (weight - mean) / std
        output = F.conv3d(x, standardized_weight, self.bias, self.stride,
                          self.padding, self.dilation, self.groups)
        output = output * self.gain
        return output

# --- Helper Modules for UNet ---

class ResidualBlock3D(nn.Module):
    """
    Residual block with two WSConv3D layers, GroupNorm, SiLU, and time embedding.
    """
    def __init__(self, in_channels: int, out_channels: int, *, 
                 time_emb_dim: int = None, groups: int = 8, dropout: float = 0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        safe_groups_out = groups
        if out_channels > 0 :
            if groups > out_channels or out_channels % groups != 0:
                for g in range(min(groups, out_channels), 0, -1):
                    if out_channels % g == 0:
                        safe_groups_out = g
                        break
                if out_channels % safe_groups_out != 0: 
                    safe_groups_out = 1 
                if groups != safe_groups_out and groups > 1: 
                    logging.debug(f"ResBlock: Adjusted GroupNorm groups from {groups} to {safe_groups_out} for out_channels={out_channels}")
        elif out_channels == 0: 
             safe_groups_out = 1 
        else: 
            raise ValueError(f"out_channels cannot be negative, got {out_channels}")


        self.conv1 = WSConv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups=safe_groups_out, num_channels=out_channels) if out_channels > 0 else nn.Identity()
        self.act1 = nn.SiLU()
        
        self.conv2 = WSConv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=safe_groups_out, num_channels=out_channels) if out_channels > 0 else nn.Identity()
        self.act2 = nn.SiLU()

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if time_emb_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, out_channels * 2 if out_channels > 0 else 2) 
            )
        else:
            self.time_mlp = None

        self.residual_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor = None) -> torch.Tensor:
        h = x
        h = self.conv1(h)
        if self.out_channels > 0: 
            h = self.norm1(h)
        
        if self.time_mlp is not None and t_emb is not None and self.out_channels > 0:
            time_encoding = self.time_mlp(t_emb) 
            time_encoding = time_encoding.view(time_encoding.shape[0], -1, 1, 1, 1) 
            scale, shift = time_encoding.chunk(2, dim=1) 
            h = h * (scale + 1) + shift

        h = self.act1(h)
        h = self.dropout(h)
        h = self.conv2(h)
        if self.out_channels > 0:
            h = self.norm2(h)
        h = self.act2(h)
        
        return h + self.residual_conv(x)


class AttentionBlock3D(nn.Module):
    """
    Self-attention block for 3D feature maps.
    """
    def __init__(self, channels: int, num_heads: int = 8, head_dim: int = None, groups: int = 8):
        super().__init__()
        if channels <= 0: 
            self.channels = 0
            self.identity = nn.Identity()
            logging.debug("AttentionBlock3D initialized with 0 channels, will act as Identity.")
            return
            
        self.channels = channels
        self.num_heads = num_heads # Store original requested num_heads
        if head_dim is None:
            current_num_heads = num_heads
            if channels % current_num_heads != 0 :
                if current_num_heads > channels and channels > 0 : current_num_heads = channels # Cap heads at channels
                while channels % current_num_heads != 0 and current_num_heads > 1:
                    current_num_heads -=1
                if channels % current_num_heads != 0: 
                    current_num_heads = 1
                if num_heads != current_num_heads: # Log if changed
                    logging.warning(f"AttentionBlock3D: channels ({channels}) not divisible by num_heads ({num_heads}). Adjusted num_heads to {current_num_heads}.")
                self.num_heads = current_num_heads # Update to adjusted
            head_dim = channels // self.num_heads
        
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        
        safe_groups = groups
        if groups > channels or channels % groups != 0:
            for g in range(min(groups, channels), 0, -1):
                if channels % g == 0:
                    safe_groups = g
                    break
            if channels % safe_groups != 0: safe_groups = 1
            if groups != safe_groups and groups > 1 :
                logging.debug(f"Attention: Adjusted GroupNorm groups from {groups} to {safe_groups} for channels={channels}")


        self.norm = nn.GroupNorm(num_groups=safe_groups, num_channels=channels)
        self.to_qkv = nn.Conv3d(channels, self.num_heads * self.head_dim * 3, kernel_size=1, bias=False)
        self.to_out = nn.Conv3d(self.num_heads * self.head_dim, channels, kernel_size=1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor = None) -> torch.Tensor:
        if self.channels == 0:
            return self.identity(x)
            
        B, C, X, Y, Z = x.shape
        res_x = x
        x_norm = self.norm(x)
        
        qkv = self.to_qkv(x_norm).chunk(3, dim=1)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q.view(B, self.num_heads, self.head_dim, X * Y * Z)
        k = k.view(B, self.num_heads, self.head_dim, X * Y * Z)
        v = v.view(B, self.num_heads, self.head_dim, X * Y * Z)

        attention_scores = torch.einsum('b h d s, b h d t -> b h s t', q, k) * self.scale
        attention_probs = F.softmax(attention_scores, dim=-1)
        out = torch.einsum('b h s t, b h d t -> b h d s', attention_probs, v)
        out = out.reshape(B, self.num_heads * self.head_dim, X, Y, Z)
        out = self.to_out(out)
        return out + res_x


class UNetBlock(nn.Module):
    """
    A UNet building block: sequence of ResidualBlock3D and optionally AttentionBlock3D.
    """
    def __init__(self, in_channels: int, out_channels: int, *, 
                 time_emb_dim: int, num_internal_residual_blocks: int = 1, 
                 use_attention: bool = False, attention_heads: int = 8, 
                 dropout: float = 0.0, groups: int = 8):
        super().__init__()
        self.res_blocks = nn.ModuleList()
        current_res_block_in_channels = in_channels
        for _ in range(num_internal_residual_blocks):
            self.res_blocks.append(
                ResidualBlock3D(current_res_block_in_channels, out_channels, 
                                time_emb_dim=time_emb_dim, dropout=dropout, groups=groups)
            )
            current_res_block_in_channels = out_channels 
        
        self.attn = AttentionBlock3D(out_channels, num_heads=attention_heads, groups=groups) if use_attention and out_channels > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        for res_block in self.res_blocks:
            x = res_block(x, t_emb)
        x = self.attn(x)
        return x

class Downsample3D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = WSConv3d(channels, channels, kernel_size=3, stride=2, padding=1) if channels > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor = None) -> torch.Tensor:
        if isinstance(self.conv, nn.Identity): return x
        return self.conv(x)

class Upsample3D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(channels, channels, kernel_size=4, stride=2, padding=1) if channels > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor = None) -> torch.Tensor:
        if isinstance(self.conv_transpose, nn.Identity): return x
        return self.conv_transpose(x)

# --- Main UNet3D Model ---

class UNet3D(nn.Module):
    def __init__(self,
                 input_channels: int, 
                 model_channels: int,
                 output_channels: int, 
                 channel_mults: tuple = (1, 2, 4, 8),
                 num_residual_blocks_per_stage: int = 2, 
                 time_embedding_dim: int = 128,
                 time_mlp_hidden_dim: int = 512,
                 time_final_emb_dim: int = None,
                 attention_resolutions_indices: tuple = (1,), 
                 attention_heads: int = 8,
                 dropout: float = 0.0,
                 groups: int = 8,
                 initial_conv_kernel_size: int = 7 
                 ):
        super().__init__()

        self.total_input_channels = input_channels 
        self.output_channels = output_channels 

        time_final_emb_dim = time_final_emb_dim if time_final_emb_dim is not None else model_channels * 4

        self.model_channels = model_channels
        self.channel_mults = channel_mults
        self.num_residual_blocks_per_stage = num_residual_blocks_per_stage
        self.attention_resolutions_indices = attention_resolutions_indices
        self.attention_heads = attention_heads
        self.dropout = dropout
        self.groups = groups

        self.time_embedder = FullTimeEmbedding(
            fourier_embedding_dim=time_embedding_dim,
            mlp_hidden_dim=time_mlp_hidden_dim,
            final_embedding_dim=time_final_emb_dim
        )
        
        padding_init = initial_conv_kernel_size // 2
        self.init_conv = WSConv3d(self.total_input_channels, model_channels, 
                                  kernel_size=initial_conv_kernel_size, padding=padding_init)

        # --- Encoder ---
        self.down_blocks = nn.ModuleList()
        current_block_input_ch = model_channels 
        ch_list = [current_block_input_ch] 

        for i_stage, mult in enumerate(channel_mults):
            stage_output_channels = model_channels * mult 
            use_attn_this_stage = (i_stage in attention_resolutions_indices)
            
            for k_block_in_stage in range(num_residual_blocks_per_stage):
                self.down_blocks.append(
                    UNetBlock( 
                        in_channels=current_block_input_ch, 
                        out_channels=stage_output_channels, 
                        time_emb_dim=time_final_emb_dim,
                        num_internal_residual_blocks=1, 
                        use_attention=use_attn_this_stage if k_block_in_stage == num_residual_blocks_per_stage - 1 and stage_output_channels > 0 else False,
                        attention_heads=attention_heads,
                        dropout=dropout,
                        groups=groups
                    )
                )
                current_block_input_ch = stage_output_channels 
            
            ch_list.append(current_block_input_ch) 
            
            if i_stage != len(channel_mults) - 1:
                self.down_blocks.append(Downsample3D(current_block_input_ch))
        
        # --- Middle ---
        self.middle_block1 = UNetBlock(
            current_block_input_ch, current_block_input_ch, time_emb_dim=time_final_emb_dim,
            num_internal_residual_blocks=1, use_attention=True if current_block_input_ch > 0 else False, 
            attention_heads=attention_heads,
            dropout=dropout, groups=groups
        )
        self.middle_block2 = UNetBlock( 
            current_block_input_ch, current_block_input_ch, time_emb_dim=time_final_emb_dim,
            num_internal_residual_blocks=1, use_attention=False, 
            dropout=dropout, groups=groups
        )

        # --- Decoder ---
        self.up_blocks = nn.ModuleList()
        for i_stage, mult in reversed(list(enumerate(channel_mults))):
            stage_target_out_channels = model_channels * mult 
            use_attn_this_stage = (i_stage in attention_resolutions_indices)
            
            encoder_skip_ch_for_this_stage = ch_list.pop() 

            for k_block_in_stage in range(num_residual_blocks_per_stage):
                input_to_unet_block = (current_block_input_ch + encoder_skip_ch_for_this_stage) if k_block_in_stage == 0 else current_block_input_ch
                
                self.up_blocks.append(
                    UNetBlock(
                        in_channels=input_to_unet_block,
                        out_channels=stage_target_out_channels,
                        time_emb_dim=time_final_emb_dim,
                        num_internal_residual_blocks=1,
                        use_attention=use_attn_this_stage if k_block_in_stage == num_residual_blocks_per_stage - 1 and stage_target_out_channels > 0 else False,
                        attention_heads=attention_heads,
                        dropout=dropout,
                        groups=groups
                    )
                )
                current_block_input_ch = stage_target_out_channels 
            
            if i_stage != 0: 
                self.up_blocks.append(Upsample3D(current_block_input_ch))
        
        assert len(ch_list) == 1, f"ch_list should have 1 element left (from init_conv), but has {len(ch_list)}"
        
        final_layer_in_channels = current_block_input_ch 
        self.final_norm = nn.GroupNorm(num_groups=min(groups, final_layer_in_channels if final_layer_in_channels > 0 else 1), num_channels=final_layer_in_channels) if final_layer_in_channels > 0 else nn.Identity()
        self.final_act = nn.SiLU()
        self.final_conv = nn.Conv3d(final_layer_in_channels, self.output_channels, kernel_size=1) if final_layer_in_channels > 0 else nn.Identity()


    def forward(self, x_combined: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the UNet3D.
        Args:
            x_combined (torch.Tensor): The combined input tensor (e.g., x_t + x_self_cond or just x_t).
                                       Shape (B, C_total_input, X, Y, Z).
            time (torch.Tensor): Time step tensor. Shape (B,).
        Returns:
            torch.Tensor: Output tensor. Shape (B, C_output, X, Y, Z).
        """
        if x_combined.shape[1] != self.total_input_channels:
            # This check should ideally not be hit if main.py calculates total_input_channels correctly
            # based on diffusion_config.self_condition_diffusion_process
            logging.error(f"Input x_combined has {x_combined.shape[1]} channels, "
                             f"but model was initialized with total_input_channels={self.total_input_channels}.")
            # Fallback or raise error:
            # For robustness, if shapes mismatch, try to process x_combined as is, but it might fail later.
            # Or, if x_combined has fewer channels than expected, it means self-cond was expected but not provided correctly.
            # If it has more, it's a more severe mismatch.
            # For now, let it proceed and potentially fail in init_conv if shapes are truly incompatible.
            # A better approach would be to ensure config and main.py are perfectly aligned.

        t_emb = self.time_embedder(time)
        h = self.init_conv(x_combined) 
        
        encoder_skips_features = [h] 
        temp_h = h
        down_block_module_idx = 0 

        for i_level_encoder, _ in enumerate(self.channel_mults):
            for _ in range(self.num_residual_blocks_per_stage):
                temp_h = self.down_blocks[down_block_module_idx](temp_h, t_emb)
                down_block_module_idx += 1
            encoder_skips_features.append(temp_h) 
            if i_level_encoder != len(self.channel_mults) - 1:
                temp_h = self.down_blocks[down_block_module_idx](temp_h) 
                down_block_module_idx +=1
        
        h_middle = self.middle_block1(temp_h, t_emb)
        h_middle = self.middle_block2(h_middle, t_emb)
        
        temp_h_decoder = h_middle
        up_block_module_idx = 0 

        for i_level_decoder, _ in reversed(list(enumerate(self.channel_mults))):
            skip_h_from_encoder = encoder_skips_features.pop() 
            
            for k_block_in_decoder_stage in range(self.num_residual_blocks_per_stage):
                if k_block_in_decoder_stage == 0: 
                    if temp_h_decoder.shape[2:] != skip_h_from_encoder.shape[2:]:
                        temp_h_decoder = F.interpolate(temp_h_decoder, size=skip_h_from_encoder.shape[2:], mode='trilinear', align_corners=False)
                        logging.debug(f"Decoder: Resized temp_h_decoder from {temp_h_decoder.shape} to {skip_h_from_encoder.shape} for skip concat.")
                    temp_h_decoder = torch.cat([temp_h_decoder, skip_h_from_encoder], dim=1)
                    
                temp_h_decoder = self.up_blocks[up_block_module_idx](temp_h_decoder, t_emb)
                up_block_module_idx +=1
            
            if i_level_decoder != 0:
                temp_h_decoder = self.up_blocks[up_block_module_idx](temp_h_decoder) 
                up_block_module_idx +=1
        
        if isinstance(self.final_norm, nn.Identity): 
            out = temp_h_decoder
        else:
            out = self.final_norm(temp_h_decoder)
        out = self.final_act(out)
        if isinstance(self.final_conv, nn.Identity):
            # This case means final_layer_in_channels was 0, which is problematic.
            # The output should always have self.output_channels.
            # If final_conv is Identity, it implies temp_h_decoder already has self.output_channels
            # and final_layer_in_channels matched self.output_channels.
            # However, if final_layer_in_channels was 0, temp_h_decoder has 0 channels.
            # This should be prevented by ensuring model_channels and channel_mults don't lead to 0.
            if temp_h_decoder.shape[1] != self.output_channels:
                 logging.warning(f"Final conv is Identity, but input channels {temp_h_decoder.shape[1]} != output_channels {self.output_channels}")
            out = temp_h_decoder # This might be an issue if channels don't match self.output_channels
        else:
            out = self.final_conv(out)
        return out

# --- Example Usage (for testing the model structure) ---
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO) # Changed from DEBUG for less verbose default
    logging.info("Testing UNet3D model structure (simplified input)...")

    batch_size = 2
    bits_xt_channels = 5 
    bits_self_cond_channels = 5 
    use_self_cond_test = True

    total_unet_input_channels = bits_xt_channels
    if use_self_cond_test:
        total_unet_input_channels += bits_self_cond_channels
        
    unet_output_channels = bits_xt_channels

    dim_x, dim_y, dim_z = 16, 16, 16

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    dummy_combined_input = torch.randn(batch_size, total_unet_input_channels, dim_x, dim_y, dim_z).to(device)
    dummy_time = torch.rand(batch_size).to(device)

    try:
        unet_model = UNet3D(
            input_channels=total_unet_input_channels, 
            model_channels=32,      
            output_channels=unet_output_channels, 
            channel_mults=(1, 2, 2), 
            num_residual_blocks_per_stage=1, 
            time_embedding_dim=64, 
            time_mlp_hidden_dim=128,
            time_final_emb_dim=128,  
            attention_resolutions_indices=(1,), 
            attention_heads=4,
            dropout=0.0, 
            groups=8, 
            initial_conv_kernel_size=3
        ).to(device)

        logging.info(f"UNet3D model instantiated successfully with total_input_channels={total_unet_input_channels}.")
        num_params = sum(p.numel() for p in unet_model.parameters() if p.requires_grad)
        logging.info(f"Number of trainable parameters: {num_params / 1e6:.2f} M")

        logging.info("Performing a test forward pass...")
        with torch.no_grad():
            output = unet_model(dummy_combined_input, dummy_time)
        
        logging.info(f"Output shape: {output.shape}")
        expected_output_shape = (batch_size, unet_output_channels, dim_x, dim_y, dim_z)
        assert output.shape == expected_output_shape, \
            f"Output shape {output.shape} does not match expected shape {expected_output_shape}"
        logging.info("Test forward pass completed successfully.")

    except Exception as e:
        logging.error(f"Error during UNet3D test: {e}", exc_info=True)
        raise 
