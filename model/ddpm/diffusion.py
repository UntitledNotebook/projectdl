# diffusion.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from tqdm.auto import tqdm 
import numpy as np 

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

class BitDiffusion(nn.Module):
    def __init__(self, 
                 model, 
                 analog_bit_scale=1.0, 
                 self_condition_enabled_in_model=True, # This now refers to the diffusion process behavior
                 gamma_ns=0.0002, 
                 gamma_ds=0.00025 
                 ):
        super().__init__()
        self.model = model # UNet3D model, expects combined input if self_condition_enabled_in_model is True
        self.analog_bit_scale = analog_bit_scale
        self.self_condition_enabled_in_model = self_condition_enabled_in_model # Behavior flag for BitDiffusion
        self.gamma_ns = gamma_ns
        self.gamma_ds = gamma_ds

        logging.info(f"BitDiffusion initialized with: analog_bit_scale={analog_bit_scale}, "
                     f"self_condition_enabled_in_model (process behavior)={self_condition_enabled_in_model}, "
                     f"gamma_ns={gamma_ns}, gamma_ds={gamma_ds}")

    def _gamma(self, t: torch.Tensor) -> torch.Tensor:
        return torch.cos(((t + self.gamma_ns) / (1.0 + self.gamma_ds)) * math.pi / 2)**2

    def q_sample(self, x_start_bits: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start_bits)
        gamma_t_batch = self._gamma(t) 
        view_shape = (-1, *([1] * (x_start_bits.ndim - 1)))
        sqrt_gamma_t = torch.sqrt(gamma_t_batch).view(view_shape)
        sqrt_one_minus_gamma_t = torch.sqrt(torch.clamp(1.0 - gamma_t_batch, min=0.0) + 1e-8).view(view_shape)
        xt = sqrt_gamma_t * x_start_bits + sqrt_one_minus_gamma_t * noise
        return xt

    def p_losses(self, x_start_bits: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start_bits)
        xt = self.q_sample(x_start_bits, t, noise=noise) 

        x0_self_cond_for_concat = torch.zeros_like(x_start_bits) 
        model_input_combined = xt # Default input if no self-conditioning

        if self.self_condition_enabled_in_model:
            batch_size = xt.shape[0]
            # Determine if we generate a self-conditioning term for this batch
            use_model_pred_for_sc_mask = (torch.rand(batch_size, device=xt.device) > 0.5).view(batch_size, *([1] * (xt.ndim - 1))).float()
            
            if torch.any(use_model_pred_for_sc_mask > 0): 
                with torch.no_grad():
                    # Model call to generate the self-conditioning term.
                    # Input is xt concatenated with zeros (as per paper's f(xt, 0, t))
                    input_for_sc_generation = torch.cat([xt, torch.zeros_like(x_start_bits)], dim=1)
                    predicted_x0_for_sc_generation = self.model(input_for_sc_generation, t)
                
                # Apply this prediction only to the samples selected by the mask
                x0_self_cond_for_concat = predicted_x0_for_sc_generation.detach() * use_model_pred_for_sc_mask
            
            # Concatenate for the final model prediction
            model_input_combined = torch.cat([xt, x0_self_cond_for_concat], dim=1)
        
        # Model predicts x0 based on the (potentially combined) input
        x0_predicted_final = self.model(model_input_combined, t)
        loss = F.mse_loss(x0_predicted_final, x_start_bits) # Loss is against the original x_start_bits
        return loss

    def _ddim_step(self, 
                   xt: torch.Tensor, 
                   x0_pred: torch.Tensor, 
                   t_now: torch.Tensor, 
                   t_next: torch.Tensor) -> torch.Tensor:
        x0_pred_clipped = torch.clamp(x0_pred, -self.analog_bit_scale, self.analog_bit_scale)
        gamma_now_batch = self._gamma(t_now)    
        gamma_next_batch = self._gamma(t_next)  
        view_shape = (-1, *([1] * (xt.ndim - 1)))
        sqrt_gamma_now = torch.sqrt(gamma_now_batch).view(view_shape)
        sqrt_one_minus_gamma_now = torch.sqrt(torch.clamp(1.0 - gamma_now_batch, min=0.0) + 1e-8).view(view_shape)
        eps_pred = (xt - sqrt_gamma_now * x0_pred_clipped) / (sqrt_one_minus_gamma_now + 1e-8) # Added epsilon for stability
        sqrt_gamma_next = torch.sqrt(gamma_next_batch).view(view_shape)
        sqrt_one_minus_gamma_next = torch.sqrt(torch.clamp(1.0 - gamma_next_batch, min=0.0) + 1e-8).view(view_shape)
        x_next_xt = sqrt_gamma_next * x0_pred_clipped + sqrt_one_minus_gamma_next * eps_pred
        return x_next_xt

    @torch.no_grad()
    def sample(self, 
               batch_size: int, 
               shape: tuple,    
               device: torch.device, 
               num_steps: int, 
               time_difference_td: float = 0.0,
               x_true_bits: torch.Tensor = None, 
               mask: torch.Tensor = None         
               ) -> torch.Tensor:
        is_inpainting = x_true_bits is not None and mask is not None
        
        if is_inpainting:
            if x_true_bits.shape[0] != mask.shape[0]: raise ValueError("Batch size mismatch for inpainting.")
            batch_size = x_true_bits.shape[0] 
            if x_true_bits.shape[1:] != shape: raise ValueError(f"Shape mismatch for inpainting.")
            x_true_bits = x_true_bits.to(device)
            mask = mask.to(device)
            if not ( (mask.ndim == x_true_bits.ndim and mask.shape[1] == 1) or (mask.shape == x_true_bits.shape) ):
                raise ValueError(f"Mask shape error for inpainting.")
            logging.info(f"Starting DDIM inpainting with {num_steps} steps, td={time_difference_td}...")
        else:
            logging.info(f"Starting DDIM unconditional sampling with {num_steps} steps, td={time_difference_td}...")

        if is_inpainting:
            initial_noise_for_unknown = torch.randn((batch_size, *shape), device=device)
            t_initial = torch.ones(batch_size, device=device)
            x_true_noised_to_t1 = self.q_sample(x_true_bits, t_initial)
            current_xt = (1.0 - mask) * initial_noise_for_unknown + mask * x_true_noised_to_t1
        else:
            current_xt = torch.randn((batch_size, *shape), device=device) 
        
        current_x0_self_cond_for_concat = torch.zeros_like(current_xt) # This has C_xt channels
        times = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
        final_predicted_x0 = None 

        for i in tqdm(range(num_steps), desc=f"DDIM {'Inpainting' if is_inpainting else 'Sampling'} Progress"):
            t_next_val = max(times[i+1] - time_difference_td / num_steps, 0.0)
            t_now_batch = torch.full((batch_size,), times[i], device=device, dtype=torch.float32)
            t_next_batch = torch.full((batch_size,), t_next_val, device=device, dtype=torch.float32)
 
            model_input_combined_sampling = current_xt
            if self.self_condition_enabled_in_model:
                model_input_combined_sampling = torch.cat([current_xt, current_x0_self_cond_for_concat], dim=1)
            
            predicted_x0_this_step = self.model(model_input_combined_sampling, t_now_batch)

            if self.self_condition_enabled_in_model:
                current_x0_self_cond_for_concat = predicted_x0_this_step.detach() 

            if i == num_steps - 1: 
                final_predicted_x0 = predicted_x0_this_step
            
            x_prev_candidate = self._ddim_step(current_xt, predicted_x0_this_step, t_now_batch, t_next_batch)

            if is_inpainting:
                if t_next_val > 1e-5: 
                    x_true_noised_to_t_next = self.q_sample(x_true_bits, t_next_batch) # Use same t_next
                    current_xt = (1.0 - mask) * x_prev_candidate + mask * x_true_noised_to_t_next
                else: 
                    current_xt = (1.0 - mask) * x_prev_candidate + mask * x_true_bits
            else: 
                current_xt = x_prev_candidate
            
        output_analog_bits = final_predicted_x0 if not is_inpainting and final_predicted_x0 is not None else current_xt
        
        return output_analog_bits
    
def get_diffusion(model: nn.Module, config: dict[str, any]) -> BitDiffusion:
    return BitDiffusion(
        model=model,
        analog_bit_scale=config.get("analog_bit_scale", 1.0),
        self_condition_enabled_in_model=config.get("self_condition_diffusion_process", True),
        gamma_ns=config.get("gamma_ns", 0.0002),
        gamma_ds=config.get("gamma_ds", 0.00025)
    )

# --- Example Usage (Illustrative) ---
if __name__ == '__main__':
    class DummyUNet3D(nn.Module): 
        def __init__(self, total_input_channels, output_channels_x0, time_emb_dim_dummy): # Simplified
            super().__init__()
            self.total_input_channels = total_input_channels
            self.output_channels_x0 = output_channels_x0
            
            self.main_path = nn.Sequential(
                nn.Conv3d(self.total_input_channels, 32, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv3d(32, self.output_channels_x0, kernel_size=3, padding=1)
            )
            logging.info(f"DummyUNet3D initialized: total_input_channels={self.total_input_channels}, "
                         f"output_channels_x0={self.output_channels_x0}")

        def forward(self, x_combined, raw_time_t): # x_combined is already concatenated
            return self.main_path(x_combined)

    logging.info("Testing BitDiffusion components (with simplified UNet input)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_bit_len = 6 # Channels for x_t, and also for x_self_cond if used
    
    # Determine total UNet input channels based on self-conditioning for the diffusion process
    diffusion_uses_self_cond = True # Test with self-conditioning in diffusion process
    
    unet_total_input_ch = data_bit_len
    if diffusion_uses_self_cond:
        unet_total_input_ch += data_bit_len # Add channels for x_self_cond

    dummy_model_instance = DummyUNet3D(
        total_input_channels=unet_total_input_ch, 
        output_channels_x0=data_bit_len, # Model predicts original x0 channels
        time_emb_dim_dummy=128 
    ).to(device)

    diffusion_process = BitDiffusion(
        model=dummy_model_instance,
        analog_bit_scale=1.0,
        self_condition_enabled_in_model=diffusion_uses_self_cond # This flag controls BitDiffusion's behavior
    )

    batch_size_test = 2
    spatial_shape_tuple = (data_bit_len, 8, 8, 8) # C_xt, X, Y, Z
    dummy_x0_analog = torch.randn(batch_size_test, *spatial_shape_tuple, device=device) * diffusion_process.analog_bit_scale
    dummy_t_for_loss = torch.rand(batch_size_test, device=device) 
    
    loss = diffusion_process.p_losses(dummy_x0_analog, dummy_t_for_loss)
    logging.info(f"Calculated p_losses: {loss.item()}")
    assert loss.ndim == 0

    num_sampling_steps = 10 
    logging.info(f"\nStarting UNCONDITIONAL sample generation for {num_sampling_steps} steps...")
    generated_samples_uncond = diffusion_process.sample(
        batch_size=batch_size_test, 
        shape=spatial_shape_tuple, # Shape of x_t
        device=device,
        num_steps=num_sampling_steps,
        time_difference_td=0.1 
    )
    logging.info(f"Generated unconditional samples shape: {generated_samples_uncond.shape}")
    assert generated_samples_uncond.shape == (batch_size_test, *spatial_shape_tuple)
    
    logging.info(f"\nStarting INPAINTING test for {num_sampling_steps} steps...")
    mask_np = np.ones((batch_size_test, 1, spatial_shape_tuple[1], spatial_shape_tuple[2], spatial_shape_tuple[3]), dtype=np.float32) 
    x_len, y_len, z_len = spatial_shape_tuple[1], spatial_shape_tuple[2], spatial_shape_tuple[3]
    mask_np[:, :, x_len//4:x_len*3//4, y_len//4:y_len*3//4, z_len//4:z_len*3//4] = 0.0 
    dummy_mask = torch.from_numpy(mask_np).to(device)
    dummy_x_true_for_inpaint = dummy_x0_analog.clone().to(device)

    inpainted_samples = diffusion_process.sample(
        batch_size=None, 
        shape=spatial_shape_tuple, # Shape of x_t
        x_true_bits=dummy_x_true_for_inpaint,
        mask=dummy_mask,
        device=device,
        num_steps=num_sampling_steps,
        time_difference_td=0.05
    )
    logging.info(f"Inpainted samples shape: {inpainted_samples.shape}")
    assert inpainted_samples.shape == (batch_size_test, *spatial_shape_tuple)

    known_part_original = dummy_x_true_for_inpaint * dummy_mask
    known_part_inpainted = inpainted_samples * dummy_mask
    assert torch.allclose(known_part_inpainted, known_part_original, atol=1e-4), \
        "Known regions in inpainted output do not match x_true_bits."
    logging.info("Inpainting test: Known regions correctly preserved.")

    logging.info("BitDiffusion tests completed.")

