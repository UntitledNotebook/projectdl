import torch
import numpy as np
from typing import Dict, Any, Optional
import os
import logging
from .data import get_processed_block_embedding_table, map_embeddings_to_ids

def create_ema_decay_schedule(config: Dict[str, Any]) -> callable:
    """
    Creates a schedule for Exponential Moving Average (EMA) decay rate.
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
        
        param_data_for_ema = param.data.to(ema_param.dtype)
        ema_param.data.mul_(decay).add_(param_data_for_ema, alpha=1.0 - decay)

@torch.no_grad()
def copy_params_to_ema(model: torch.nn.Module, ema_model: torch.nn.Module) -> None:
    """Copies parameters from the training model to the EMA model."""
    for param, ema_param in zip(model.parameters(), ema_model.parameters()):
        if not ema_param.is_leaf or not param.is_leaf:
            continue
        ema_param.data.copy_(param.data.to(ema_param.dtype))

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    checkpoint_dir: str,
    ema_model: Optional[torch.nn.Module] = None,
    filename_prefix: str = "checkpoint"
) -> None:
    """
    Saves training state to checkpoint directory.
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
    
    try:
        torch.save(save_data, checkpoint_path_step)
        torch.save(save_data, latest_checkpoint_path)
        logging.debug(f"Saved checkpoint to {checkpoint_path_step} and {latest_checkpoint_path}")
    except Exception as e:
        logging.error(f"Error saving checkpoint at step {step}: {e}")

def save_samples_as_block_ids(
    generated_sample_embeddings: torch.Tensor,
    embedding_path_for_reference: str,
    embedding_dim_for_reference: int,
    model_outputs_standardized_embeddings: bool, 
    output_path_npy: str,
    metric_for_id_mapping: str = 'euclidean'
) -> None:
    """
    Converts generated sample embeddings to block IDs and saves them as a .npy file.

    Args:
        generated_sample_embeddings (torch.Tensor): Embeddings output by the model.
                                                    Shape: (N, EmbeddingDim, D, H, W).
        embedding_path_for_reference (str): Path to the original .npy file of all block embeddings.
        embedding_dim_for_reference (int): The dimension of the embeddings in the reference file.
        model_outputs_standardized_embeddings (bool): True if the generated_sample_embeddings
                                                      are in a standardized space (e.g. zero mean, unit variance).
                                                      If True, the reference embedding table will also be standardized
                                                      before finding the closest IDs.
        output_path_npy (str): Path to save the resulting block IDs .npy file.
        metric_for_id_mapping (str): Metric to use for mapping embeddings to IDs ('cosine' or 'euclidean').
    """
    logging.debug(f"Converting {generated_sample_embeddings.shape[0]} generated samples to block IDs.")
    
    try:
        reference_embedding_table = get_processed_block_embedding_table(
            embedding_path=embedding_path_for_reference,
            embedding_dim=embedding_dim_for_reference,
            standardize_table=model_outputs_standardized_embeddings,
            device=torch.device('cpu') 
        )
        logging.debug(f"Loaded reference embedding table. Shape: {reference_embedding_table.shape}, Standardized: {model_outputs_standardized_embeddings}")

        block_ids_np = map_embeddings_to_ids(
            predicted_embeddings_batch=generated_sample_embeddings,
            reference_embedding_table=reference_embedding_table,
            metric=metric_for_id_mapping
        )
        logging.debug(f"Converted embeddings to block IDs. Shape: {block_ids_np.shape}")

        os.makedirs(os.path.dirname(output_path_npy), exist_ok=True)
        np.save(output_path_npy, block_ids_np)
        logging.info(f"Saved {block_ids_np.shape[0]} samples as block IDs to: {output_path_npy}")

    except Exception as e:
        logging.error(f"Error in save_samples_as_block_ids: {e}")
        logging.error("Failed to save samples as block IDs. Saving raw embeddings instead as fallback.")
        fallback_path = output_path_npy.replace(".npy", "_raw_embeddings.npy")
        try:
            np.save(fallback_path, generated_sample_embeddings.cpu().numpy())
            logging.info(f"Saved raw sample embeddings to: {fallback_path}")
        except Exception as fallback_e:
            logging.error(f"Failed to save raw embeddings as fallback: {fallback_e}")
