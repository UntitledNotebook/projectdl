import torch
from typing import Dict, Any, Optional
import os
import logging

def create_ema_decay_schedule(config: Dict[str, Any]) -> callable:
    """
    Creates a schedule for Exponential Moving Average (EMA) decay rate.

    Args:
        config (Dict[str, Any]): Configuration dictionary. Expected keys:
            'update_after_step', 'update_every', 'inv_gamma', 
            'power', 'min_value', 'beta'.

    Returns:
        callable: A function that takes the current step and returns EMA decay.
    """
    update_after_step = config['update_after_step']
    update_every = config['update_every']
    inv_gamma = config['inv_gamma']
    power = config['power']
    min_value = config['min_value']
    beta = config['beta']

    def ema_decay_schedule(step: int) -> float:
        if step < update_after_step:
            return 0.0
        count = (step - update_after_step) // update_every
        decay = 1.0 - (1.0 + count / inv_gamma) ** -power
        current_decay = max(min_value, decay)
        current_decay = min(current_decay, beta)
        return current_decay
        
    return ema_decay_schedule

@torch.no_grad()
def apply_ema_decay(model: torch.nn.Module, ema_model: torch.nn.Module, decay: float) -> None:
    """
    Updates Exponential Moving Average (EMA) parameters.
    ema_param = decay * ema_param + (1 - decay) * param.
    """
    for param, ema_param in zip(model.parameters(), ema_model.parameters()):
        if not ema_param.is_leaf or not param.is_leaf:
             continue
        
        param_data_for_ema = param.data.to(ema_param.dtype) # Ensure dtype consistency
        ema_param.data.mul_(decay).add_(param_data_for_ema, alpha=1.0 - decay)

@torch.no_grad()
def copy_params_to_ema(model: torch.nn.Module, ema_model: torch.nn.Module) -> None:
    """Copies parameters from the training model to the EMA model."""
    for param, ema_param in zip(model.parameters(), ema_model.parameters()):
        if not ema_param.is_leaf or not param.is_leaf:
            continue
        ema_param.data.copy_(param.data.to(ema_param.dtype)) # Ensure dtype consistency

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    checkpoint_dir: str,
    ema_model: Optional[torch.nn.Module] = None,
    # scaler argument removed
    filename_prefix: str = "checkpoint"
) -> None:
    """
    Saves training state to checkpoint directory.
    Creates a step-specific checkpoint and overwrites a 'latest' checkpoint.
    """
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_path_step = os.path.join(checkpoint_dir, f"{filename_prefix}_step_{step}.pth")
    latest_checkpoint_path = os.path.join(checkpoint_dir, f"{filename_prefix}_latest.pth")

    save_data = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    if ema_model is not None:
        save_data['ema_model_state_dict'] = ema_model.state_dict()
    
    # GradScaler state saving removed
    # save_data['grad_scaler_state_dict'] = None 

    try:
        torch.save(save_data, checkpoint_path_step)
        torch.save(save_data, latest_checkpoint_path)
        logging.debug(f"Saved checkpoint to {checkpoint_path_step} and {latest_checkpoint_path}")
    except Exception as e:
        logging.error(f"Error saving checkpoint at step {step}: {e}")

