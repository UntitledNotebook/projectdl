import jax
import jax.numpy as jnp
import numpy as np
import wandb
import os
from flax import traverse_util
from typing import Dict, Any

def l2_loss(pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    """Computes L2 (mean squared error) loss.

    Args:
        pred (jnp.ndarray): Predictions, shape (batch_size, ...).
        target (jnp.ndarray): Targets, shape (batch_size, ...).

    Returns:
        jnp.ndarray: L2 loss per sample, shape (batch_size,).
    """
    return jnp.mean((pred - target) ** 2, axis=tuple(range(1, pred.ndim)))

def l1_loss(pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    """Computes L1 (mean absolute error) loss.

    Args:
        pred (jnp.ndarray): Predictions, shape (batch_size, ...).
        target (jnp.ndarray): Targets, shape (batch_size, ...).

    Returns:
        jnp.ndarray: L1 loss per sample, shape (batch_size,).
    """
    return jnp.mean(jnp.abs(pred - target), axis=tuple(range(1, pred.ndim)))

def create_ema_decay_schedule(config: Dict[str, Any]) -> callable:
    """Creates a schedule for EMA decay rate.

    Args:
        config (Dict[str, Any]): Configuration with keys:
            - update_after_step (int): Step to start EMA updates.
            - update_every (int): Frequency of EMA updates.
            - inv_gamma (float): Inverse gamma for decay schedule.
            - power (float): Power for decay schedule.
            - min_value (float): Minimum decay value.
            - beta (float): Base EMA decay rate.

    Returns:
        callable: Function that takes step and returns decay rate.
    """
    def ema_decay_schedule(step):
        if step < config['update_after_step']:
            return 0.0
        count = (step - config['update_after_step']) // config['update_every']
        decay = 1 - (1 + count / config['inv_gamma']) ** -config['power']
        decay = max(decay, config['min_value'])
        decay = min(decay, config['beta'])
        return decay
    return ema_decay_schedule

def apply_ema_decay(state: Any, decay: float) -> Any:
    """Updates EMA parameters using the specified decay rate.

    Args:
        state (Any): Training state with params and params_ema.
        decay (float): EMA decay rate.

    Returns:
        Any: Updated training state with new params_ema.
    """
    def update_ema(param, ema_param):
        return decay * ema_param + (1 - decay) * param
    params_ema = jax.tree_map(update_ema, state.params, state.params_ema)
    return state.replace(params_ema=params_ema)

def copy_params_to_ema(state: Any) -> Any:
    """Copies current parameters to EMA parameters.

    Args:
        state (Any): Training state with params and params_ema.

    Returns:
        Any: Updated training state with params copied to params_ema.
    """
    return state.replace(params_ema=state.params)

def save_checkpoint(state: Any, checkpoint_dir: str) -> None:
    """Saves training state to checkpoint directory.

    Args:
        state (Any): Training state to save.
        checkpoint_dir (str): Directory to save checkpoints.
    """
    from flax.training import checkpoints
    if jax.process_index() == 0:
        checkpoints.save_checkpoint(checkpoint_dir, jax.device_get(state), state.step, keep=3, overwrite=True)