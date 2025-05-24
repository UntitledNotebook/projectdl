import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple, Optional, Callable
from tqdm import tqdm # For progress bars in sampling

def get_ddpm_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Computes DDPM (Denoising Diffusion Probabilistic Models) parameters.

    Args:
        config (Dict[str, Any]): Configuration dictionary containing:
            - timesteps (int): Total number of diffusion steps.
            - beta_schedule (str): Type of beta schedule ('linear' or 'cosine').
            - beta_start (float, optional): Start value for linear beta schedule.
            - beta_end (float, optional): End value for linear beta schedule.
            - cosine_s (float, optional): Offset for cosine beta schedule.
            - p2_loss_weight_gamma (float, optional): Gamma for p2 loss weighting.
            - p2_loss_weight_k (float, optional): K for p2 loss weighting.
            - device (str or torch.device, optional): Device for tensor allocation.

    Returns:
        Dict[str, Any]: DDPM parameters (betas, alphas, etc.) as PyTorch tensors.
    """
    timesteps = config['timesteps']
    device = config.get('device', 'cpu')

    if config['beta_schedule'] == 'linear':
        beta_start = config.get('beta_start', 0.0001)
        beta_end = config.get('beta_end', 0.02)
        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32, device=device)
    elif config['beta_schedule'] == 'cosine':
        s = config.get('cosine_s', 0.008)
        steps = timesteps + 1
        x_time = torch.linspace(0, timesteps, steps, dtype=torch.float32, device=device)
        alphas_cumprod = torch.cos(((x_time / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1. - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clip(betas, 0.0001, 0.9999)
    else:
        raise ValueError(f"Unsupported beta_schedule: {config['beta_schedule']}")

    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod + 1e-8) 

    p2_loss_weight_gamma = config.get('p2_loss_weight_gamma', 0.0)
    p2_loss_weight_k = config.get('p2_loss_weight_k', 1.0)
    p2_loss_weight = (p2_loss_weight_k + alphas_cumprod / (1. - alphas_cumprod + 1e-8)) ** -p2_loss_weight_gamma
    p2_loss_weight = p2_loss_weight.to(device)

    return {
        'betas': betas, 'alphas': alphas, 'alphas_cumprod': alphas_cumprod,
        'alphas_cumprod_prev': alphas_cumprod_prev, 'sqrt_recip_alphas': sqrt_recip_alphas,
        'sqrt_alphas_cumprod': sqrt_alphas_cumprod,
        'sqrt_one_minus_alphas_cumprod': sqrt_one_minus_alphas_cumprod,
        'posterior_variance': posterior_variance, 'p2_loss_weight': p2_loss_weight,
        'timesteps': timesteps,
    }

def _reshape_tensor_for_broadcast(tensor_to_reshape: torch.Tensor, target_tensor: torch.Tensor) -> torch.Tensor:
    """Reshapes a 1D tensor for broadcasting with a target tensor."""
    while len(tensor_to_reshape.shape) < len(target_tensor.shape):
        tensor_to_reshape = tensor_to_reshape.unsqueeze(-1)
    return tensor_to_reshape.to(target_tensor.device)

def q_sample(x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor, ddpm_params: Dict[str, Any]) -> torch.Tensor:
    """
    Forward diffusion process: q(x_t | x_0).
    x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
    """
    sqrt_alphas_cumprod_t = ddpm_params['sqrt_alphas_cumprod'].gather(0, t)
    sqrt_one_minus_alphas_cumprod_t = ddpm_params['sqrt_one_minus_alphas_cumprod'].gather(0, t)

    sqrt_alphas_cumprod_t = _reshape_tensor_for_broadcast(sqrt_alphas_cumprod_t, x_start)
    sqrt_one_minus_alphas_cumprod_t = _reshape_tensor_for_broadcast(sqrt_one_minus_alphas_cumprod_t, x_start)

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

def noise_to_x0(noise_pred: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor, ddpm_params: Dict[str, Any]) -> torch.Tensor:
    """
    Computes predicted x_0 from predicted noise:
    x_0_pred = (x_t - sqrt(1 - alpha_bar_t) * noise_pred) / sqrt(alpha_bar_t)
    """
    sqrt_alphas_cumprod_t = ddpm_params['sqrt_alphas_cumprod'].gather(0, t)
    sqrt_one_minus_alphas_cumprod_t = ddpm_params['sqrt_one_minus_alphas_cumprod'].gather(0, t)

    sqrt_alphas_cumprod_t = _reshape_tensor_for_broadcast(sqrt_alphas_cumprod_t, noise_pred)
    sqrt_one_minus_alphas_cumprod_t = _reshape_tensor_for_broadcast(sqrt_one_minus_alphas_cumprod_t, noise_pred)
    
    return (x_t - sqrt_one_minus_alphas_cumprod_t * noise_pred) / (sqrt_alphas_cumprod_t + 1e-8)


def x0_to_noise(x0_pred: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor, ddpm_params: Dict[str, Any]) -> torch.Tensor:
    """
    Computes predicted noise from predicted x_0:
    noise_pred = (x_t - sqrt(alpha_bar_t) * x_0_pred) / sqrt(1 - alpha_bar_t)
    """
    sqrt_alphas_cumprod_t = ddpm_params['sqrt_alphas_cumprod'].gather(0, t)
    sqrt_one_minus_alphas_cumprod_t = ddpm_params['sqrt_one_minus_alphas_cumprod'].gather(0, t)

    sqrt_alphas_cumprod_t = _reshape_tensor_for_broadcast(sqrt_alphas_cumprod_t, x0_pred)
    sqrt_one_minus_alphas_cumprod_t = _reshape_tensor_for_broadcast(sqrt_one_minus_alphas_cumprod_t, x0_pred)
    
    return (x_t - sqrt_alphas_cumprod_t * x0_pred) / (sqrt_one_minus_alphas_cumprod_t + 1e-8)


def get_posterior_mean_variance(
    x_t: torch.Tensor, t: torch.Tensor, x0_pred: torch.Tensor, ddpm_params: Dict[str, Any]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes mean and variance of the posterior q(x_{t-1} | x_t, x_0_pred).
    """
    t_long = t.long()
    betas_t = ddpm_params['betas'].gather(0, t_long)
    alphas_t = ddpm_params['alphas'].gather(0, t_long)
    alphas_cumprod_t = ddpm_params['alphas_cumprod'].gather(0, t_long)
    alphas_cumprod_prev_t = ddpm_params['alphas_cumprod_prev'].gather(0, t_long)
    posterior_variance_t = ddpm_params['posterior_variance'].gather(0, t_long)

    betas_t = _reshape_tensor_for_broadcast(betas_t, x_t)
    alphas_t = _reshape_tensor_for_broadcast(alphas_t, x_t)
    alphas_cumprod_t = _reshape_tensor_for_broadcast(alphas_cumprod_t, x_t)
    alphas_cumprod_prev_t = _reshape_tensor_for_broadcast(alphas_cumprod_prev_t, x_t)
    posterior_variance_t = _reshape_tensor_for_broadcast(posterior_variance_t, x_t)

    posterior_mean_coef1 = torch.sqrt(alphas_cumprod_prev_t) * betas_t / (1. - alphas_cumprod_t + 1e-8)
    posterior_mean_coef2 = torch.sqrt(alphas_t) * (1. - alphas_cumprod_prev_t) / (1. - alphas_cumprod_t + 1e-8)
    
    posterior_mean = posterior_mean_coef1 * x0_pred + posterior_mean_coef2 * x_t
    posterior_log_variance_clipped = torch.log(torch.maximum(posterior_variance_t, torch.tensor(1e-20, device=x_t.device)))
    
    return posterior_mean, posterior_variance_t, posterior_log_variance_clipped

def p_loss(
    model: torch.nn.Module, x_start: torch.Tensor, t: torch.Tensor, ddpm_params: Dict[str, Any],
    loss_type: str = 'l2', self_condition: bool = False, pred_x0: bool = False,
    p2_loss_weight_gamma: float = 0.0
) -> torch.Tensor:
    """Computes the diffusion model training loss."""
    device = x_start.device
    noise = torch.randn_like(x_start, device=device)
    x_noisy = q_sample(x_start=x_start, t=t, noise=noise, ddpm_params=ddpm_params)

    # --- Self-Conditioning Preliminary Prediction ---
    # This part always happens if self_condition is True, to get a candidate x0_self_cond
    # The model must always receive its expected input channels.
    prelim_x0_candidate = None
    if self_condition:
        with torch.no_grad():
            # Input for preliminary prediction: x_noisy + dummy_zeros for the self-cond channel
            dummy_x0_for_prelim_pred = torch.zeros_like(x_noisy, device=device)
            prelim_model_input = torch.cat([x_noisy, dummy_x0_for_prelim_pred], dim=1)
            
            model_output_sc = model(prelim_model_input, t) 
            
            if pred_x0: # If model directly predicts x0
                 prelim_x0_candidate = model_output_sc.detach() 
            else: # If model predicts noise
                 prelim_x0_candidate = noise_to_x0(model_output_sc, x_noisy, t, ddpm_params).detach()

    # --- Prepare Input for Main Prediction (with probabilistic self-conditioning) ---
    x_model_input = x_noisy 
    if self_condition:
        # Probabilistic choice: 50% use the prelim_x0_candidate, 50% use zeros for self-cond channel
        if torch.rand(1).item() < 0.5 and prelim_x0_candidate is not None:
            # Use the computed prelim_x0_candidate
            x_model_input = torch.cat([x_noisy, prelim_x0_candidate], dim=1)
            # logging.debug("p_loss: Using PREDICTED x0_self_cond for main prediction.")
        else:
            # Use zeros for the self-conditioning channel
            dummy_x0_for_main_pred = torch.zeros_like(x_noisy, device=device)
            x_model_input = torch.cat([x_noisy, dummy_x0_for_main_pred], dim=1)
            # logging.debug("p_loss: Using ZEROS for x0_self_cond for main prediction.")
    # If not self_condition, x_model_input remains x_noisy (base channels).
    # This assumes train.py sets the model's in_channels correctly based on the global SELF_CONDITION.
    
    model_output = model(x_model_input, t)

    # --- Calculate Loss ---
    if pred_x0:
        target = x_start
        loss_val = model_output # model_output is predicted x0
    else:
        target = noise
        loss_val = model_output # model_output is predicted noise

    if loss_type == 'l2':
        loss = F.mse_loss(loss_val, target, reduction='none')
    elif loss_type == 'l1':
        loss = F.l1_loss(loss_val, target, reduction='none')
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")

    loss = loss.view(loss.shape[0], -1).mean(dim=1) # Loss per sample

    if p2_loss_weight_gamma > 0:
        p2_lw = ddpm_params['p2_loss_weight'].gather(0, t.long()).squeeze()
        loss = loss * p2_lw
    
    return loss.mean() # Average loss over the batch

def model_predict(
    model: torch.nn.Module, x: torch.Tensor, t: torch.Tensor, ddpm_params: Dict[str, Any],
    self_condition: bool = False, x0_self_cond: Optional[torch.Tensor] = None, pred_x0: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Makes a prediction with the model (either noise or x0) and returns both.
    Handles self-conditioning input preparation.
    """
    x_model_input = x 
    if self_condition: # Model expects doubled channels if self_condition is active
        if x0_self_cond is not None:
            x_model_input = torch.cat([x, x0_self_cond], dim=1)
        else:
            # First pass of self-conditioning, x0_self_cond is unknown, use zeros
            dummy_x0 = torch.zeros_like(x, device=x.device)
            x_model_input = torch.cat([x, dummy_x0], dim=1)
    # If not self_condition, x_model_input is just x (base channels).
    # This relies on the U-Net's in_channels being set correctly in train.py.

    model_output = model(x_model_input, t)

    if pred_x0:
        pred_x0_tensor = model_output
        pred_noise_tensor = x0_to_noise(pred_x0_tensor, x, t, ddpm_params) # x is the original noisy input
    else: 
        pred_noise_tensor = model_output
        pred_x0_tensor = noise_to_x0(pred_noise_tensor, x, t, ddpm_params) # x is the original noisy input
    
    return pred_x0_tensor, pred_noise_tensor

def ddpm_sample_step(
    model: torch.nn.Module, x: torch.Tensor, t: torch.Tensor, ddpm_params: Dict[str, Any],
    self_condition: bool = False, 
    x0_last: Optional[torch.Tensor] = None, 
    pred_x0: bool = False,
    clip_denoised_value: Optional[float] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Performs one step of the DDPM reverse sampling process."""
    device = x.device
    with torch.no_grad():
        pred_x0_tensor, _ = model_predict(
            model, x, t, ddpm_params,
            self_condition=self_condition, 
            x0_self_cond=x0_last, 
            pred_x0=pred_x0
        )

    if clip_denoised_value is not None:
        pred_x0_tensor = torch.clamp(pred_x0_tensor, -clip_denoised_value, clip_denoised_value)

    posterior_mean, _, posterior_log_variance_clipped = get_posterior_mean_variance(
        x_t=x, t=t, x0_pred=pred_x0_tensor, ddpm_params=ddpm_params
    )

    noise_sample = torch.randn_like(x, device=device)
    mask_t_gt_0 = (t > 0).float().view(-1, *([1]*(len(x.shape)-1)))
    noise_sample = noise_sample * mask_t_gt_0
    
    pred_x_prev = posterior_mean + torch.exp(0.5 * posterior_log_variance_clipped) * noise_sample
    return pred_x_prev, pred_x0_tensor


def ddpm_inpaint_sample_step(
    model: torch.nn.Module, x: torch.Tensor, t: torch.Tensor, ddpm_params: Dict[str, Any],
    x_true_masked: torch.Tensor, mask: torch.Tensor,
    self_condition: bool = False, 
    x0_last: Optional[torch.Tensor] = None, 
    pred_x0: bool = False,
    clip_denoised_value: Optional[float] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Performs one step of DDPM inpainting sampling (RePaint variant)."""
    device = x.device
    with torch.no_grad():
        pred_x0_tensor, _ = model_predict(
            model, x, t, ddpm_params,
            self_condition=self_condition, 
            x0_self_cond=x0_last,
            pred_x0=pred_x0
        )

    if clip_denoised_value is not None:
        pred_x0_tensor = torch.clamp(pred_x0_tensor, -clip_denoised_value, clip_denoised_value)

    posterior_mean_unknown, _, posterior_log_var_unknown = get_posterior_mean_variance(
        x_t=x, t=t, x0_pred=pred_x0_tensor, ddpm_params=ddpm_params
    )
    noise_sample_unknown = torch.randn_like(x, device=device)
    mask_t_gt_0 = (t > 0).float().view(-1, *([1]*(len(x.shape)-1)))
    noise_sample_unknown = noise_sample_unknown * mask_t_gt_0
    x_prev_unknown_region = posterior_mean_unknown + torch.exp(0.5 * posterior_log_var_unknown) * noise_sample_unknown

    if torch.any(t > 0):
        t_minus_1 = torch.maximum(t - 1, torch.zeros_like(t))
        noise_for_known_diffusion = torch.randn_like(x_true_masked, device=device)
        x_prev_known_region = q_sample(
            x_start=x_true_masked, t=t_minus_1, noise=noise_for_known_diffusion, ddpm_params=ddpm_params
        )
        pred_x_prev = x_prev_known_region * mask + x_prev_unknown_region * ~mask
    else: 
        pred_x_prev = x_true_masked * mask + pred_x0_tensor * ~mask
        
    return pred_x_prev, pred_x0_tensor


@torch.no_grad()
def sample_loop(
    shape: Tuple[int, ...], timesteps: int, device: torch.device,
    sampler_fn: Callable[..., Tuple[torch.Tensor, torch.Tensor]], 
    initial_voxel: Optional[torch.Tensor] = None,
    progress_desc: str = "sampling loop", 
    **sampler_fn_kwargs 
) -> torch.Tensor:
    """
    Generic sampling loop for diffusion models.
    The `sampler_fn` is expected to be a functools.partial object that has
    `model`, `ddpm_params`, `pred_x0`, and `self_condition` pre-filled.
    """
    batch_size = shape[0]
    voxel = torch.randn(shape, device=device) if initial_voxel is None else initial_voxel.to(device)
    
    x0_pred_for_next_step = None 

    for i in tqdm(reversed(range(timesteps)), desc=progress_desc, total=timesteps, dynamic_ncols=True):
        t_tensor = torch.full((batch_size,), i, device=device, dtype=torch.long)
        
        current_x0_last = x0_pred_for_next_step 
        
        voxel, current_x0_pred = sampler_fn(
            x=voxel, 
            t=t_tensor, 
            x0_last=current_x0_last, 
            **sampler_fn_kwargs
        )
        
        x0_pred_for_next_step = current_x0_pred.detach()
            
    return voxel
