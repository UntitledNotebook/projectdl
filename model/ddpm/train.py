import torch
import torch.optim as optim
import numpy as np
import os
import logging
import functools # For functools.partial
from tqdm import tqdm
from typing import Any, Dict, Tuple, Optional

from accelerate import Accelerator
from accelerate.utils import set_seed

from .model import UNet3D
from .data import get_dataset 
from .diffusion import p_loss, get_ddpm_params, sample_loop, ddpm_sample_step
from .utils import create_ema_decay_schedule, apply_ema_decay, copy_params_to_ema, save_checkpoint, save_samples_as_block_ids

# --- Global Configurations ---
MAX_STEP = 100
DATA_PATH = "data/block_ids_32_32.npy"
EMBEDDING_PATH = "output/block2vec/block_embeddings.npy"
OUTPUT_DIR = "output/ddpm_minecraft_accelerate" 
CHECKPOINT_DIR = os.path.abspath(os.path.join(OUTPUT_DIR, "checkpoints"))
SAMPLE_DIR = os.path.abspath(os.path.join(OUTPUT_DIR, "samples"))

SEED = 42
BATCH_SIZE = 32 # This will be the per-device batch size
GRADIENT_ACCUMULATION_STEPS = 1 # Set > 1 to simulate larger batch size
EFFECTIVE_BATCH_SIZE = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS

VOXEL_SHAPE = (32, 32, 32)
EMBEDDING_DIM = 4
STANDARDIZE_EMBEDDINGS = True

# Model Config
MODEL_DIM = 64
DIM_MULTS = (1, 2, 4, 8)
UNET_OUT_CHANNELS = EMBEDDING_DIM

# Diffusion Config
BETA_SCHEDULE = 'cosine'
TIMESTEPS = 1000
SELF_CONDITION = False
PRED_X0 = False
CLIP_DENOISED_VALUE = None

# Loss Config
LOSS_TYPE = 'l2'
P2_LOSS_WEIGHT_GAMMA = 0.5 
P2_LOSS_WEIGHT_K = 1.0     

# Optimizer Config
LEARNING_RATE = 1e-4
BETA1 = 0.9
BETA2 = 0.99
EPS = 1e-8

# EMA Config
EMA_UPDATE_AFTER_STEP = 10
EMA_UPDATE_EVERY = 10 # Should be a multiple of GRADIENT_ACCUMULATION_STEPS if used with it
EMA_INV_GAMMA = 1.0
EMA_POWER = 0.75
EMA_MIN_VALUE = 0.0
EMA_BETA = 0.9999

# Logging & Saving Config
LOG_EVERY_STEPS = 10 # Log actual optimizer steps
SAVE_AND_SAMPLE_EVERY = 10 # Save actual optimizer steps
NUM_SAMPLE = 4

# WANDB (Optional)
WANDB_LOG_TRAIN = True 
WANDB_PROJECT = 'projectdl_accelerate'
WANDB_JOB_TYPE = 'train_3d_unet_accelerate'
WANDB_GROUP = 'ddpm_accelerate'
WANDB_TAGS = ['pytorch', '3d_diffusion', BETA_SCHEDULE, f'p2_gamma_{P2_LOSS_WEIGHT_GAMMA}', f'p2_k_{P2_LOSS_WEIGHT_K}']
if SELF_CONDITION: WANDB_TAGS.append('self_cond')
if PRED_X0: WANDB_TAGS.append('pred_x0')

# DEVICE will be handled by Accelerator

def get_wandb_config() -> Dict[str, Any]:
    """Creates a dictionary of hyperparameters for WandB logging."""
    return {
        "MAX_STEP": MAX_STEP,
        "DATA_PATH": DATA_PATH,
        "EMBEDDING_PATH": EMBEDDING_PATH,
        "OUTPUT_DIR": OUTPUT_DIR,
        "SEED": SEED,
        "BATCH_SIZE_PER_DEVICE": BATCH_SIZE,
        "GRADIENT_ACCUMULATION_STEPS": GRADIENT_ACCUMULATION_STEPS,
        "EFFECTIVE_BATCH_SIZE": EFFECTIVE_BATCH_SIZE,
        "VOXEL_SHAPE": VOXEL_SHAPE,
        "EMBEDDING_DIM": EMBEDDING_DIM,
        "STANDARDIZE_EMBEDDINGS": STANDARDIZE_EMBEDDINGS,
        "MODEL_DIM": MODEL_DIM,
        "DIM_MULTS": DIM_MULTS,
        "UNET_OUT_CHANNELS": UNET_OUT_CHANNELS,
        "BETA_SCHEDULE": BETA_SCHEDULE,
        "TIMESTEPS": TIMESTEPS,
        "SELF_CONDITION": SELF_CONDITION,
        "PRED_X0": PRED_X0,
        "CLIP_DENOISED_VALUE": CLIP_DENOISED_VALUE,
        "LOSS_TYPE": LOSS_TYPE,
        "P2_LOSS_WEIGHT_GAMMA": P2_LOSS_WEIGHT_GAMMA,
        "P2_LOSS_WEIGHT_K": P2_LOSS_WEIGHT_K,
        "LEARNING_RATE": LEARNING_RATE,
        "BETA1": BETA1,
        "BETA2": BETA2,
        "EPS": EPS,
        "EMA_UPDATE_AFTER_STEP": EMA_UPDATE_AFTER_STEP,
        "EMA_UPDATE_EVERY": EMA_UPDATE_EVERY,
        "EMA_INV_GAMMA": EMA_INV_GAMMA,
        "EMA_POWER": EMA_POWER,
        "EMA_MIN_VALUE": EMA_MIN_VALUE,
        "EMA_BETA": EMA_BETA,
        "LOG_EVERY_STEPS": LOG_EVERY_STEPS,
        "SAVE_AND_SAMPLE_EVERY": SAVE_AND_SAMPLE_EVERY,
        "NUM_SAMPLE": NUM_SAMPLE,
        # "DEVICE": str(DEVICE) # Accelerator handles device
    }

def create_model_and_optimizer() -> Tuple[torch.nn.Module, torch.optim.Optimizer]:
    # No device argument needed here, Accelerator will handle placement
    unet_in_channels_main_path = EMBEDDING_DIM * 2 if SELF_CONDITION else EMBEDDING_DIM
    
    model = UNet3D(
        dim=MODEL_DIM,
        dim_mults=DIM_MULTS,
        in_channels=unet_in_channels_main_path, 
        out_dim=UNET_OUT_CHANNELS,    
        resnet_block_groups=8,        
        use_linear_attention=True,    
        attn_heads=4,                 
        attn_dim_head=32              
    ) # .to(device) removed

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2), eps=EPS)
    return model, optimizer

def train():
    global WANDB_LOG_TRAIN # Allow modification if wandb import fails

    # Initialize Accelerator
    accelerator = Accelerator(gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS)

    logging.basicConfig(
        level=logging.INFO if accelerator.is_main_process else logging.ERROR, # Log more on main process
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    if WANDB_LOG_TRAIN and accelerator.is_main_process:
        try:
            import wandb
            wandb_config = get_wandb_config()
            wandb_config["DEVICE"] = str(accelerator.device) # Log the device Accelerator chose
            wandb.init(project=WANDB_PROJECT, job_type=WANDB_JOB_TYPE, group=WANDB_GROUP, tags=WANDB_TAGS, config=wandb_config)
        except ImportError:
            logging.warning("wandb not installed, skipping wandb logging.")
            WANDB_LOG_TRAIN = False # Disable for all processes if import fails
        
    if accelerator.is_main_process:
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(SAMPLE_DIR, exist_ok=True)
    
    set_seed(SEED) # Use Accelerate's set_seed

    logging.info(f"Using device: {accelerator.device}")
    logging.info(f"Effective batch size: {EFFECTIVE_BATCH_SIZE} (Per-device: {BATCH_SIZE} * Grad Accum: {GRADIENT_ACCUMULATION_STEPS})")
    logging.info(f"Self-conditioning globally enabled: {SELF_CONDITION}")
    logging.info(f"Model predicting x0: {PRED_X0}")
    logging.info(f"P2 Loss Weighting: k={P2_LOSS_WEIGHT_K}, gamma={P2_LOSS_WEIGHT_GAMMA}")

    dataloader = get_dataset(
        data_path=DATA_PATH, embedding_path=EMBEDDING_PATH, batch_size=BATCH_SIZE,
        voxel_shape=VOXEL_SHAPE, embedding_dim=EMBEDDING_DIM,
        standardize_embeddings=STANDARDIZE_EMBEDDINGS, num_workers=0, shuffle=True # num_workers=0 often safer
    )

    model, optimizer = create_model_and_optimizer() 
    
    unet_in_channels_ema_main_path = EMBEDDING_DIM * 2 if SELF_CONDITION else EMBEDDING_DIM
    # EMA model is not prepared with accelerator if used only for eval/saving on main process
    ema_model = UNet3D(
        dim=MODEL_DIM, dim_mults=DIM_MULTS, in_channels=unet_in_channels_ema_main_path,
        out_dim=UNET_OUT_CHANNELS, resnet_block_groups=8, use_linear_attention=True,
        attn_heads=4, attn_dim_head=32
    ).to(accelerator.device) # Move EMA model to device manually if needed for sampling
    copy_params_to_ema(model, ema_model) # model is not yet prepared, so this is fine
    ema_model.eval()
    # print the parameters of the EMA model
    logging.info(f"EMA model parameters: {sum(p.numel() for p in ema_model.parameters())}")

    # Prepare components with Accelerator
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)


    start_step = 0
    latest_checkpoint_path = os.path.join(CHECKPOINT_DIR, "checkpoint_latest.pth")
    if os.path.exists(latest_checkpoint_path):
        if accelerator.is_main_process:
            logging.info(f"Restoring checkpoint from {latest_checkpoint_path}")
        try:
            # Checkpoint loading needs to be handled carefully with Accelerator
            # For single GPU, direct loading might work, but for multi-GPU, accelerator.load_state is preferred.
            # For simplicity here, we'll keep direct loading, assuming single GPU for now.
            # Ensure that the model is loaded *before* accelerator.prepare if not using accelerator.load_state
            # However, since we prepare then load, we need to unwrap for state_dict loading.
            
            # To load state, it's often easier to load before prepare, or use accelerator.load_state
            # For this example, let's assume we load into the raw model then prepare.
            # This part might need adjustment based on exact checkpoint saving/loading strategy with accelerate.
            # A common pattern is to load state into the raw model, then prepare.
            # If loading after prepare, you'd load into accelerator.unwrap_model(model).
            
            # Let's refine this: create model, load state, then prepare.
            # So, the prepare call will be after this block.
            # For now, this is a placeholder and might need to be moved before `accelerator.prepare`
            # or use `accelerator.load_state(CHECKPOINT_DIR)`
            
            # Simplified loading for single GPU (adjust for more complex setups)
            # This assumes the checkpoint was saved from an unwrapped model.
            unwrapped_model_for_load = accelerator.unwrap_model(model) 
            checkpoint_data = torch.load(latest_checkpoint_path, map_location=accelerator.device)
            unwrapped_model_for_load.load_state_dict(checkpoint_data['model_state_dict'])
            optimizer.load_state_dict(checkpoint_data['optimizer_state_dict']) # Optimizer state can be loaded after prepare
            
            # EMA model loading (already on device)
            ema_model.load_state_dict(checkpoint_data['ema_model_state_dict'])
            start_step = checkpoint_data['step'] + 1
            if accelerator.is_main_process:
                logging.info(f"Restored checkpoint. Starting from step {start_step}")
        except Exception as e:
            if accelerator.is_main_process:
                logging.error(f"Could not load checkpoint: {e}. Starting from scratch.")
            start_step = 0


    ddpm_params_config = {
        'beta_schedule': BETA_SCHEDULE, 
        'timesteps': TIMESTEPS,
        'p2_loss_weight_gamma': P2_LOSS_WEIGHT_GAMMA, 
        'p2_loss_weight_k': P2_LOSS_WEIGHT_K,         
        'device': accelerator.device # Use accelerator's device
    }
    if BETA_SCHEDULE == 'linear':
        ddpm_params_config['beta_start'] = 0.0001 
        ddpm_params_config['beta_end'] = 0.02   
    elif BETA_SCHEDULE == 'cosine':
        ddpm_params_config['cosine_s'] = 0.008 
    ddpm_params = get_ddpm_params(ddpm_params_config)


    ema_decay_fn = create_ema_decay_schedule({
        'update_after_step': EMA_UPDATE_AFTER_STEP, 'update_every': EMA_UPDATE_EVERY,
        'inv_gamma': EMA_INV_GAMMA, 'power': EMA_POWER,
        'min_value': EMA_MIN_VALUE, 'beta': EMA_BETA
    })
    
    completed_optimizer_steps = start_step
    progress_bar = tqdm(
        initial=completed_optimizer_steps, 
        total=MAX_STEP, 
        desc="Training Optimizer Steps", 
        disable=not accelerator.is_main_process, # Disable progress bar on non-main processes
        dynamic_ncols=True
    )
    
    
    global_step_counter = completed_optimizer_steps * GRADIENT_ACCUMULATION_STEPS # tracks forward passes

    while completed_optimizer_steps < MAX_STEP:
        model.train() # Set model to train mode each iteration (or epoch)
        for batch_idx, x_start in enumerate(dataloader):
            # x_start is already on accelerator.device due to accelerator.prepare(dataloader)
            
            t_rand = torch.randint(0, ddpm_params['timesteps'], (x_start.shape[0],), device=accelerator.device).long()

            # Loss calculation happens inside the accumulation context
            with accelerator.accumulate(model):
                loss = p_loss(model, x_start, t_rand, ddpm_params,
                              loss_type=LOSS_TYPE, self_condition=SELF_CONDITION,
                              pred_x0=PRED_X0, 
                              p2_loss_weight_gamma=P2_LOSS_WEIGHT_GAMMA)
                
                accelerator.backward(loss) # Use accelerator for backward pass
                
                # if accelerator.sync_gradients:
                #    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                optimizer.zero_grad()

            global_step_counter += 1

            # Check if an optimizer step was performed (after accumulation)
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_optimizer_steps += 1

                unwrapped_model = accelerator.unwrap_model(model)
                if completed_optimizer_steps >= EMA_UPDATE_AFTER_STEP and completed_optimizer_steps % EMA_UPDATE_EVERY == 0:
                    current_ema_decay = ema_decay_fn(completed_optimizer_steps)
                    apply_ema_decay(unwrapped_model, ema_model, current_ema_decay)

                # Logging (based on optimizer steps)
                if completed_optimizer_steps % LOG_EVERY_STEPS == 0:
                    log_data = {'step': completed_optimizer_steps, 'loss': loss.item() / GRADIENT_ACCUMULATION_STEPS} # Avg loss over accumulation
                    if accelerator.is_main_process:
                        # logging.info(f"Step: {completed_optimizer_steps}, Avg Loss: {loss.item() / GRADIENT_ACCUMULATION_STEPS:.4f}")
                        if WANDB_LOG_TRAIN:
                            wandb.log(log_data, step=completed_optimizer_steps)
                
                # Save and Sample (based on optimizer steps)
                if completed_optimizer_steps > 0 and completed_optimizer_steps % SAVE_AND_SAMPLE_EVERY == 0 :
                    if accelerator.is_main_process:
                        logging.info(f"Saving checkpoint and generating samples at optimizer step {completed_optimizer_steps}...")
                        
                        unwrapped_model_to_save = accelerator.unwrap_model(model)
                        unwrapped_ema_model_to_save = ema_model # ema_model is already unwrapped and on device

                        save_checkpoint(
                            model=unwrapped_model_to_save, optimizer=optimizer, step=completed_optimizer_steps, 
                            checkpoint_dir=CHECKPOINT_DIR, ema_model=unwrapped_ema_model_to_save, 
                            filename_prefix="checkpoint"
                        )
                        
                        unwrapped_ema_model_to_save.eval() # Ensure EMA model is in eval for sampling
                        sampler_step_fn = functools.partial(
                            ddpm_sample_step, 
                            model=unwrapped_ema_model_to_save, # Use unwrapped EMA model
                            ddpm_params=ddpm_params, 
                            pred_x0=PRED_X0,
                            self_condition=SELF_CONDITION
                        )
                        
                        sampler_kwargs = {}
                        if CLIP_DENOISED_VALUE is not None:
                             sampler_kwargs['clip_denoised_value'] = CLIP_DENOISED_VALUE

                        samples = sample_loop(
                            shape=(NUM_SAMPLE, UNET_OUT_CHANNELS, *VOXEL_SHAPE),
                            timesteps=ddpm_params['timesteps'], device=accelerator.device,
                            sampler_fn=sampler_step_fn, 
                            progress_desc=f"Sampling at step {completed_optimizer_steps}", 
                            **sampler_kwargs
                        )
                        
                        sample_save_path = os.path.join(SAMPLE_DIR, f"samples_step_{completed_optimizer_steps}.npy")
                        save_samples_as_block_ids(
                            samples, EMBEDDING_PATH, EMBEDDING_DIM,
                            STANDARDIZE_EMBEDDINGS, sample_save_path
                        )
                        
                        if WANDB_LOG_TRAIN:
                            wandb.log({"samples_saved_path": sample_save_path}, step=completed_optimizer_steps)
                    accelerator.wait_for_everyone() 
            
            if completed_optimizer_steps >= MAX_STEP:
                break
        if completed_optimizer_steps >= MAX_STEP:
            break


    progress_bar.close()
    if accelerator.is_main_process:
        logging.info("Training finished.")
        if WANDB_LOG_TRAIN:
            wandb.finish()

if __name__ == '__main__':
    # To run with accelerate:
    # accelerate config (if first time, or to change settings)
    # accelerate launch train.py
    # For single GPU, it will run as a normal script but with Accelerate's benefits.
    train()
