# utils.py
import logging
import numpy as np
import torch

from data import floats_to_ids

logger = logging.getLogger(__name__)

def log_samples_to_wandb(tag_name, samples_analog_bits, bit_length, global_step, accelerator, num_to_log=4):
    """
    Converts analog bits to integer IDs and logs a few 2D slices to WandB.
    Assumes samples_analog_bits is a torch tensor on CPU.

    Args:
        tag_name (str): Name for the logged samples in WandB.
        samples_analog_bits (torch.Tensor): Tensor of analog bits from the model.
        bit_length (int): The bit length used for encoding.
        global_step (int): The current global training step.
        accelerator (Accelerator): The Hugging Face Accelerator instance.
        num_to_log (int): Number of samples from the batch to log.
    """
    if not (accelerator.is_main_process and accelerator.trackers):
        return

    try:
        import wandb
    except ImportError:
        logger.warning("wandb not installed. Skipping sample logging to WandB.")
        return

    samples_to_log = samples_analog_bits[:num_to_log].cpu() # Ensure it's on CPU
    reconstructed_ids_np = floats_to_ids(samples_to_log, bit_length) # (N, X, Y, Z)

    if reconstructed_ids_np is None or reconstructed_ids_np.size == 0:
        logger.warning("Reconstructed IDs are empty or None. Skipping sample logging.")
        return

    images_to_log = []
    for i in range(reconstructed_ids_np.shape[0]):
        if reconstructed_ids_np.ndim != 4: # Expecting (N, X, Y, Z) from floats_to_ids
            logger.error(f"Unexpected shape for reconstructed_ids_np sample {i}: {reconstructed_ids_np[i].shape}. Expected 3D for slicing.")
            continue
        
        # Log a central X-Y slice (at Z // 2)
        try:
            # Ensure dimensions are sufficient for slicing
            if reconstructed_ids_np.shape[3] < 1: # Check Z dimension
                 logger.warning(f"Sample {i} has Z dimension < 1, cannot take Z-center slice.")
                 continue

            central_xy_slice = reconstructed_ids_np[i, :, :, reconstructed_ids_np.shape[3] // 2]
            if central_xy_slice.max() > 0:
                central_xy_slice_norm = (central_xy_slice.astype(float) / central_xy_slice.max()) * 255
            else:
                central_xy_slice_norm = central_xy_slice.astype(float) # Avoid division by zero if max is 0
            central_xy_slice_uint8 = central_xy_slice_norm.astype(np.uint8)
            images_to_log.append(wandb.Image(central_xy_slice_uint8, caption=f"Sample {i} XY-Slice (Z-center)"))

            # Central X-Z slice (at Y // 2)
            if reconstructed_ids_np.shape[2] < 1: # Check Y dimension
                 logger.warning(f"Sample {i} has Y dimension < 1, cannot take Y-center slice.")
                 continue
            central_xz_slice = reconstructed_ids_np[i, :, reconstructed_ids_np.shape[2] // 2, :]
            if central_xz_slice.max() > 0:
                central_xz_slice_norm = (central_xz_slice.astype(float) / central_xz_slice.max()) * 255
            else:
                central_xz_slice_norm = central_xz_slice.astype(float)
            central_xz_slice_uint8 = central_xz_slice_norm.astype(np.uint8)
            images_to_log.append(wandb.Image(central_xz_slice_uint8, caption=f"Sample {i} XZ-Slice (Y-center)"))
        except IndexError as e:
            logger.error(f"Error slicing sample {i} for WandB logging: {e}. Sample shape: {reconstructed_ids_np[i].shape}")
            continue
            
    if images_to_log:
        wandb.log({tag_name: images_to_log}, step=global_step)
        logger.info(f"Logged {len(images_to_log)//2} samples to WandB as '{tag_name}'.")
    elif reconstructed_ids_np.shape[0] > 0 : # If there were samples but none could be logged
        logger.warning(f"No valid slices generated for WandB logging from {reconstructed_ids_np.shape[0]} samples.")

