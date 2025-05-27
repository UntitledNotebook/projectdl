# inpaint.py
import torch
import numpy as np
import os
import logging

# Import configurations and modules from your project
import config as InpaintFullConfig # Assuming your config file is config.py
from data import floats_to_ids, ids_to_floats
from model import get_model
from diffusion import BitDiffusion

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    cfg = InpaintFullConfig.inpaint_config
    data_cfg = InpaintFullConfig.data_config
    model_cfg = InpaintFullConfig.model_config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    bit_length = data_cfg["bit_representation_length"]
    image_spatial_shape = tuple(data_cfg["image_spatial_shape"]) # (X, Y, Z)
    block_ids = np.load(cfg["data_file_path"])
    assert block_ids.shape == (1, *image_spatial_shape)
    x_true_analog_bits = ids_to_floats(block_ids, bit_length).to(device) # (1, C, X, Y, Z)

    mask = torch.ones(1, 1, *image_spatial_shape, dtype=torch.float32, device=device)
    mask[:, :, 
         2 : image_spatial_shape[0] - 2,
         2 : image_spatial_shape[1] - 2,
         2 : image_spatial_shape[2] - 2] = 0.0
    logging.info(f"Created inpainting mask. Shape: {mask.shape}. ")


    # --- Load Model ---
    # Use model_config from the main config file for UNet architecture
    unet = get_model(model_cfg)
    unet.load_state_dict(torch.load(cfg["checkpoint_path"], map_location=device))
    unet.to(device)
    unet.eval()

    # --- Initialize Diffusion Process ---
    diffusion_process = BitDiffusion(
        model=unet,
        analog_bit_scale=cfg["analog_bit_scale"],
        self_condition_enabled_in_model=cfg["self_condition_diffusion_process"],
        gamma_ns=cfg["gamma_ns"],
        gamma_ds=cfg["gamma_ds"]
    )

    # --- Perform Inpainting ---
    logging.info(f"Starting inpainting for sample 0 from the dataset...")
    inpainted_analog_bits = diffusion_process.sample(
        batch_size=1, # Inpainting one sample at a time
        shape=(bit_length, *image_spatial_shape), # C, X, Y, Z
        device=device,
        num_steps=cfg["sampling_steps"],
        time_difference_td=cfg["time_difference_td"],
        x_true_bits=x_true_analog_bits, # Shape (1, C, X, Y, Z)
        mask=mask # Shape (1, 1, X, Y, Z)
    )
    logging.info(f"Inpainting finished. Output shape: {inpainted_analog_bits.shape}")

    # --- Convert to IDs and Save ---
    # floats_to_ids expects (B, C, X, Y, Z) and returns (B, X, Y, Z)
    inpainted_ids_np = floats_to_ids(inpainted_analog_bits.cpu(), bit_length) # (1, X, Y, Z)
    
    if inpainted_ids_np is None or inpainted_ids_np.size == 0:
        logging.error("Failed to convert inpainted bits to IDs.")
        return

    inpainted_ids_single_sample_np = inpainted_ids_np[0] # Get the single sample (X,Y,Z)

    os.makedirs(cfg["output_dir"], exist_ok=True)
    output_filename = os.path.join(cfg["output_dir"], f"inpainted_sample_0_steps_{cfg['sampling_steps']}.npy")
    np.save(output_filename, inpainted_ids_single_sample_np)
    logging.info(f"Saved inpainted sample (integer IDs) to {output_filename}")

    # Optional: Save a slice for quick visual check if desired
    # slice_to_save = inpainted_ids_single_sample_np[:, :, image_spatial_shape[2] // 2]
    # np.save(os.path.join(cfg["output_dir"], "inpainted_slice.npy"), slice_to_save)
    # logging.info("Saved a central Z-slice of the inpainted sample.")

if __name__ == "__main__":
    # Ensure your config.py has 'inpaint_config', 'data_config', and 'model_config'
    # Example structure for config.py:
    #
    # data_config = { ... }
    # model_config = { ... } # This should match the training config of the checkpoint
    # inpaint_config = {
    #     "data_file_path": "/path/to/full_dataset_for_selecting_a_sample.npy", # e.g., your training .npy
    #     "sampling_steps": 100,
    #     "time_difference_td": 0.0,
    #     "checkpoint_path": "/path/to/your/trained_model.pt", # or .pth
    #     "output_dir": "outputs/inpainted_results",
    #     "analog_bit_scale": 1.0, # Should match training
    #     "self_condition_diffusion_process": True, # Should match training
    #     "gamma_ns": 0.0002, # Should match training
    #     "gamma_ds": 0.00025, # Should match training
    # }
    #
    # # Then, in your actual config.py, you'd have these dictionaries defined.
    # # For this example to run, you'd need a config.py in the same directory.
    
    # Check if config has the necessary keys (basic check)
    if not hasattr(InpaintFullConfig, 'inpaint_config') or \
       not hasattr(InpaintFullConfig, 'data_config') or \
       not hasattr(InpaintFullConfig, 'model_config'):
        logging.error("config.py is missing one or more required configurations: "
                      "inpaint_config, data_config, model_config. Please define them.")
    else:
        main()
