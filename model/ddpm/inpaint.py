# inpaint.py
import torch
import numpy as np
import os
import logging

import config as InpaintFullConfig
from data import floats_to_ids, ids_to_floats
from model import get_model
from diffusion import get_diffusion
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
argparser = argparse.ArgumentParser(description="Inpaint a single sample using a pre-trained model.")
argparser.add_argument("--input_file_path", type=str, required=True, help="Path to the input .npy file with block IDs.")
argparser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the pre-trained model checkpoint.")
argparser.add_argument("--output_file_path", type=str, required=True, help="Path to save the inpainted output .npy file.")
argparser.add_argument("--sampling_steps", type=int, default=1000, help="Number of sampling steps for inpainting.")
argparser.add_argument("--time_difference_td", type=float, default=0.0, help="Time difference for the diffusion process.")
args = argparser.parse_args()

def main():
    data_cfg = InpaintFullConfig.data_config
    model_cfg = InpaintFullConfig.model_config
    diffusion_cfg = InpaintFullConfig.diffusion_config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    bit_length = data_cfg["bit_representation_length"]
    image_spatial_shape = tuple(data_cfg["image_spatial_shape"])
    block_ids = np.load(args.input_file_path)
    if block_ids.ndim == 3:
        block_ids = block_ids[np.newaxis, ...]
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
    unet.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    unet.to(device)
    unet.eval()

    # --- Initialize Diffusion Process ---
    diffusion_process = get_diffusion(unet, diffusion_cfg)

    # --- Perform Inpainting ---
    inpainted_analog_bits = diffusion_process.sample(
        batch_size=1, # Inpainting one sample at a time
        shape=(bit_length, *image_spatial_shape), # C, X, Y, Z
        device=device,
        num_steps=args.sampling_steps,
        time_difference_td=args.time_difference_td,
        x_true_bits=x_true_analog_bits, # Shape (1, C, X, Y, Z)
        mask=mask # Shape (1, 1, X, Y, Z)
    )

    # --- Convert to IDs and Save ---
    # floats_to_ids expects (B, C, X, Y, Z) and returns (B, X, Y, Z)
    inpainted_ids_np = floats_to_ids(inpainted_analog_bits.cpu(), bit_length) # (1, X, Y, Z)
    inpainted_ids_single_sample_np = inpainted_ids_np[0] # Get the single sample (X,Y,Z)

    output_dir = os.path.dirname(args.output_file_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    np.save(args.output_file_path, inpainted_ids_single_sample_np)
    logging.info(f"Saved inpainted sample (integer IDs) to {args.output_file_path}")

if __name__ == "__main__":
    main()