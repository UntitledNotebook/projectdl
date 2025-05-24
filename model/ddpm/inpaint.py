import torch
import numpy as np
import os
import logging
import functools
from typing import Tuple

try:
    from .train import (
        VOXEL_SHAPE, EMBEDDING_PATH, EMBEDDING_DIM, STANDARDIZE_EMBEDDINGS,
        MODEL_DIM, DIM_MULTS, UNET_OUT_CHANNELS, 
        BETA_SCHEDULE, TIMESTEPS, SELF_CONDITION, PRED_X0, CLIP_DENOISED_VALUE
    )
    from .model import UNet3D
    from .data import get_processed_block_embedding_table, map_ids_to_embeddings, map_embeddings_to_ids
    from .diffusion import get_ddpm_params, ddpm_inpaint_sample_step, sample_loop
except ImportError as e:
    logging.error(f"Error importing project modules. Make sure they are in the Python path: {e}")
    logging.error("You might need to run this script from the parent directory of 'train.py', etc., or adjust PYTHONPATH.")
    exit(1)


# --- Configuration for Inpainting Script ---
TARGET_VOXEL_PATH = "input_voxels/target_scene_to_inpaint.npy" # INPUT: Path to a .npy file with original block IDs (shape VOXEL_SHAPE)
INPAINTED_OUTPUT_DIR = "output/inpainted_scenes"
INPAINTED_FILENAME_PREFIX = "inpainted"

CHECKPOINT_TO_LOAD = "output/ddpm_minecraft_accelerate/checkpoints/checkpoint_latest.pth" 

MASK_METHOD = "center_square_hole" # Options: "center_square_hole", "random_patches", "bottom_half"
NUM_INPAINT_SAMPLES = 1 # How many inpainting results to generate for the target scene
BATCH_SIZE_INPAINT = 1 # Process one sample at a time for inpainting for simplicity

# Determine device (can override TRAIN_DEVICE if needed for inference)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_mask(voxel_shape: Tuple[int, int, int], method: str = "center_square_hole", 
                hole_ratio: float = 0.8, patch_ratio: float = 0.1, num_patches: int = 5) -> np.ndarray:
    """
    Creates a boolean mask for inpainting. 
    Mask is True (or 1) for known regions, False (or 0) for regions to be inpainted.

    Args:
        voxel_shape (Tuple[int, int, int]): The shape of the voxel grid (D, H, W).
        method (str): Masking strategy.
        hole_ratio (float): For 'center_square_hole', ratio of dimension to be hole.
        patch_ratio (float): For 'random_patches', ratio of dimension for each patch.
        num_patches (int): For 'random_patches', number of random patches to mask out.

    Returns:
        np.ndarray: Boolean mask of shape `voxel_shape`. True for known, False for unknown.
    """
    D, H, W = voxel_shape
    mask = np.ones(voxel_shape, dtype=bool)  # Initialize mask with all known (True)

    if method == "center_square_hole":
        d_hole, h_hole, w_hole = int(D * hole_ratio), int(H * hole_ratio), int(W * hole_ratio)
        d_start, h_start, w_start = (D - d_hole) // 2, (H - h_hole) // 2, (W - w_hole) // 2
        mask[d_start : d_start + d_hole, 
             h_start : h_start + h_hole, 
             w_start : w_start + w_hole] = False # Mark center as unknown
    elif method == "random_patches":
        patch_d, patch_h, patch_w = int(D * patch_ratio), int(H * patch_ratio), int(W * patch_ratio)
        for _ in range(num_patches):
            d_start = np.random.randint(0, D - patch_d + 1)
            h_start = np.random.randint(0, H - patch_h + 1)
            w_start = np.random.randint(0, W - patch_w + 1)
            mask[d_start : d_start + patch_d, 
                 h_start : h_start + patch_h, 
                 w_start : w_start + patch_w] = False
    elif method == "bottom_half":
        mask[D // 2 :, :, :] = False # Mask out bottom half
    elif method == "top_half":
        mask[: D // 2, :, :] = False
    else:
        logging.warning(f"Unknown mask method '{method}'. Using a fully known mask (no inpainting).")
    
    logging.info(f"Created mask with method '{method}'. Percentage of unknown region: {100 * (1 - mask.mean()):.2f}%")
    return mask


def inpaint_scene():
    """
    Main function to load data, model, create mask, perform inpainting, and save results.
    """
    os.makedirs(INPAINTED_OUTPUT_DIR, exist_ok=True)

    # 1. Load Model
    logging.info(f"Loading model checkpoint from: {CHECKPOINT_TO_LOAD}")
    if not os.path.exists(CHECKPOINT_TO_LOAD):
        logging.error(f"Checkpoint file not found: {CHECKPOINT_TO_LOAD}")
        return

    # Determine UNet input channels based on SELF_CONDITION from train.py
    unet_in_channels = EMBEDDING_DIM * 2 if SELF_CONDITION else EMBEDDING_DIM
    
    model = UNet3D(
        dim=MODEL_DIM,
        dim_mults=DIM_MULTS,
        in_channels=unet_in_channels, 
        out_dim=UNET_OUT_CHANNELS,
        resnet_block_groups=8, # Should match training config if not in globals
        use_linear_attention=True, # Should match training config
        attn_heads=4,
        attn_dim_head=32
    ).to(DEVICE)

    checkpoint_data = torch.load(CHECKPOINT_TO_LOAD, map_location=DEVICE)
    
    # Try to load EMA model state if available, otherwise regular model state
    if 'ema_model_state_dict' in checkpoint_data and checkpoint_data['ema_model_state_dict'] is not None:
        model.load_state_dict(checkpoint_data['ema_model_state_dict'])
        logging.info("Loaded EMA model state from checkpoint.")
    elif 'model_state_dict' in checkpoint_data:
        model.load_state_dict(checkpoint_data['model_state_dict'])
        logging.info("Loaded regular model state from checkpoint (EMA not found or was None).")
    else:
        logging.error("Checkpoint does not contain 'ema_model_state_dict' or 'model_state_dict'.")
        return
    model.eval()

    # 2. Load DDPM parameters
    ddpm_params_config = {
        'beta_schedule': BETA_SCHEDULE, 'timesteps': TIMESTEPS,
        'p2_loss_weight_gamma': 0.0, 'p2_loss_weight_k': 1.0, # p2 loss not used for sampling
        'device': DEVICE
    }
    if BETA_SCHEDULE == 'linear':
        ddpm_params_config['beta_start'] = 0.0001 
        ddpm_params_config['beta_end'] = 0.02   
    elif BETA_SCHEDULE == 'cosine':
        ddpm_params_config['cosine_s'] = 0.008 
    ddpm_params = get_ddpm_params(ddpm_params_config)

    # 3. Load target voxel data (original block IDs)
    logging.info(f"Loading target voxel data from: {TARGET_VOXEL_PATH}")
    if not os.path.exists(TARGET_VOXEL_PATH):
        logging.error(f"Target voxel file not found: {TARGET_VOXEL_PATH}")
        return
    target_voxel_ids_np = np.load(TARGET_VOXEL_PATH).astype(np.int64)
    if target_voxel_ids_np.shape != VOXEL_SHAPE:
        logging.error(f"Target voxel shape mismatch. Expected {VOXEL_SHAPE}, got {target_voxel_ids_np.shape}")
        return
    
    # Reshape to add batch dimension (B=1)
    target_voxel_ids_batch_np = target_voxel_ids_np[np.newaxis, ...] # (1, D, H, W)

    # 4. Prepare embedding tables
    # `embedding_table_for_input` is standardized if model expects standardized input
    # `embedding_table_for_id_map` is the original one for mapping output back to IDs
    embedding_table_for_input = get_processed_block_embedding_table(
        embedding_path=EMBEDDING_PATH,
        embedding_dim=EMBEDDING_DIM,
        standardize_table=STANDARDIZE_EMBEDDINGS, # Standardize if training data was standardized
        device=DEVICE
    )
    # For mapping model output (standardized embeddings) back to IDs, the reference table should also be standardized.
    # If model outputs non-standardized, then this reference should be non-standardized.
    # Assuming model outputs in the same space as its input (STANDARDIZE_EMBEDDINGS flag controls this)
    reference_table_for_id_mapping = get_processed_block_embedding_table(
        embedding_path=EMBEDDING_PATH,
        embedding_dim=EMBEDDING_DIM,
        standardize_table=STANDARDIZE_EMBEDDINGS, # Use standardized table if model outputs standardized
        device=torch.device('cpu') # map_embeddings_to_ids expects CPU tensor for cdist
    )


    # 5. Convert target voxel IDs to embeddings (for x_true_masked)
    # These embeddings should be in the space the model expects (e.g., standardized)
    target_voxel_embeddings_true = map_ids_to_embeddings(
        target_voxel_ids_batch_np,
        all_block_embeddings_for_input=embedding_table_for_input,
        target_device=DEVICE
    ) # Shape: (1, E, D, H, W)

    for i in range(NUM_INPAINT_SAMPLES):
        logging.info(f"Starting inpainting sample {i+1}/{NUM_INPAINT_SAMPLES}...")
        
        # 6. Create mask
        mask_np = create_mask(VOXEL_SHAPE, method=MASK_METHOD) # (D, H, W), boolean
        mask_torch = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(DEVICE) # (1, 1, D, H, W)
                                                                                   # Expand for batch and channel
        
        # x_true_masked: known regions from ground truth, others can be zero or anything as they are ignored
        x_true_masked = target_voxel_embeddings_true * mask_torch # Apply mask

        # 7. Prepare inpainting sampler function
        inpainting_sampler_fn = functools.partial(
            ddpm_inpaint_sample_step,
            model=model,
            ddpm_params=ddpm_params,
            pred_x0=PRED_X0,
            self_condition=SELF_CONDITION
        )

        # 8. Run inpainting sampling loop
        # Initial voxel for sample_loop should be random noise.
        # ddpm_inpaint_sample_step handles the combination of known/unknown parts.
        initial_noise = torch.randn_like(target_voxel_embeddings_true, device=DEVICE)
        
        inpainted_embeddings = sample_loop(
            shape=target_voxel_embeddings_true.shape, # (1, E, D, H, W)
            timesteps=ddpm_params['timesteps'],
            device=DEVICE,
            sampler_fn=inpainting_sampler_fn,
            initial_voxel=initial_noise, # Start from noise
            progress_desc=f"Inpainting Sample {i+1}",
            # Pass necessary kwargs for ddpm_inpaint_sample_step
            x_true_masked=x_true_masked,
            mask=mask_torch,
            clip_denoised_value=CLIP_DENOISED_VALUE 
        ) # Shape: (1, E, D, H, W)

        # 9. Convert inpainted embeddings back to block IDs
        # The reference_table_for_id_mapping should be in the same space as inpainted_embeddings
        inpainted_block_ids_np = map_embeddings_to_ids(
            predicted_embeddings_batch=inpainted_embeddings,
            reference_embedding_table=reference_table_for_id_mapping 
        ) # Shape: (1, D, H, W)
        
        # Squeeze batch dimension for saving
        final_inpainted_scene = inpainted_block_ids_np.squeeze(0) # (D, H, W)

        # 10. Save the result
        output_filename = f"{INPAINTED_FILENAME_PREFIX}_{MASK_METHOD}_sample_{i+1}.npy"
        output_path = os.path.join(INPAINTED_OUTPUT_DIR, output_filename)
        np.save(output_path, final_inpainted_scene)
        logging.info(f"Saved inpainted scene to: {output_path}")

    logging.info("Inpainting process completed.")


if __name__ == "__main__":
    # Configure logging
    # Set level to logging.DEBUG to see detailed U-Net logs from model.py if enabled there
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    # Example to enable debug logs from model.py:
    # logging.getLogger(__name__.split('.')[0]+".model").setLevel(logging.DEBUG)


    # --- User Action Required: Create Input Directory and File ---
    # Before running, create an 'input_voxels' directory in the same location as this script,
    # and place a 'target_scene_to_inpaint.npy' file inside it.
    # This .npy file should contain your 3D numpy array of integer block IDs
    # with the shape defined by VOXEL_SHAPE in train.py (e.g., (32, 32, 32)).
    # Example to create a dummy input file:
    if not os.path.exists(TARGET_VOXEL_PATH):
        print(f"Warning: Target voxel file not found at {TARGET_VOXEL_PATH}")
        print("Creating a dummy target voxel file for demonstration.")
        os.makedirs(os.path.dirname(TARGET_VOXEL_PATH), exist_ok=True)
        dummy_voxels = np.random.randint(0, 5, size=VOXEL_SHAPE, dtype=np.int64) # Assuming 5 block types
        np.save(TARGET_VOXEL_PATH, dummy_voxels)
        print(f"Dummy file created at {TARGET_VOXEL_PATH}. Replace it with your actual data.")
        print("Ensure EMBEDDING_PATH in train.py points to a valid embedding file for these block IDs.")

    if not os.path.exists(EMBEDDING_PATH):
         print(f"CRITICAL WARNING: Embedding file not found at {EMBEDDING_PATH} (from train.py).")
         print("This script requires the block embeddings to function.")
         print("Please ensure EMBEDDING_PATH in train.py is correct and the file exists.")
         exit(1) # Optionally exit if embeddings are critical for demo to run

    inpaint_scene()
