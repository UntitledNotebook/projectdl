import torch
import torch.optim as optim
import numpy as np
import os
import logging
import functools # For functools.partial
from tqdm import tqdm
from typing import Any, Dict, Tuple, Optional
import time

from accelerate import Accelerator
from accelerate.utils import set_seed

from .model import UNet3D
from .data import get_dataset 
from .diffusion import p_loss, get_ddpm_params, sample_loop, ddpm_sample_step
from .utils import create_ema_decay_schedule, apply_ema_decay, copy_params_to_ema, save_checkpoint, save_samples_as_block_ids

# --- Global Configurations ---
MAX_STEP = 10000 
DATA_PATH = "data/block_ids_32_32.npy"
EMBEDDING_PATH = "output/block2vec/block_embeddings.npy"
OUTPUT_DIR = f"output/ddpm/{time.strftime('%Y-%m-%d_%H-%M-%S')}" 
CHECKPOINT_DIR = os.path.abspath(os.path.join(OUTPUT_DIR, "checkpoints"))
SAMPLE_DIR = os.path.abspath(os.path.join(OUTPUT_DIR, "samples"))

SEED = 42
BATCH_SIZE = 32 
GRADIENT_ACCUMULATION_STEPS = 1 
EFFECTIVE_BATCH_SIZE = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
MAX_GRAD_NORM = 1.0 # Added for gradient clipping

VOXEL_SHAPE = (32, 32, 32)
EMBEDDING_DIM = 4
STANDARDIZE_EMBEDDINGS = False

# Model Config
MODEL_DIM = 64
DIM_MULTS = (1, 2, 4, 8)
UNET_OUT_CHANNELS = EMBEDDING_DIM

# Diffusion Config
BETA_SCHEDULE = 'cosine'
TIMESTEPS = 4000 
SELF_CONDITION = True
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
EMA_UPDATE_AFTER_STEP = 500  
EMA_UPDATE_EVERY = 100       
EMA_INV_GAMMA = 1.0
EMA_POWER = 0.75
EMA_MIN_VALUE = 0.0
EMA_BETA = 0.9999

# Logging & Saving Config 
LOG_EVERY_STEPS = 1         
SAVE_AND_SAMPLE_EVERY = 2000 
NUM_SAMPLE = 4
SAMPLE_TO_BLOCK_ID_METRIC = 'euclidean'

# WANDB (Optional)
WANDB_LOG_TRAIN = True 
WANDB_PROJECT = 'projectdl' 
WANDB_GROUP = 'ddpm' 
WANDB_TAGS = ['pytorch', '3d_diffusion', BETA_SCHEDULE, f'p2_gamma_{P2_LOSS_WEIGHT_GAMMA}', f'p2_k_{P2_LOSS_WEIGHT_K}']
if SELF_CONDITION: WANDB_TAGS.append('self_cond')
if PRED_X0: WANDB_TAGS.append('pred_x0')


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
        "MAX_GRAD_NORM": MAX_GRAD_NORM, # Added to wandb config
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
        "SAMPLE_TO_BLOCK_ID_METRIC": SAMPLE_TO_BLOCK_ID_METRIC
    }

def create_model_and_optimizer() -> Tuple[torch.nn.Module, torch.optim.Optimizer]:
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
    ) 

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2), eps=EPS)
    return model, optimizer

def train():
    global WANDB_LOG_TRAIN 

    accelerator = Accelerator(gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS)

    logging.basicConfig(
        level=logging.INFO if accelerator.is_main_process else logging.ERROR, 
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    if WANDB_LOG_TRAIN and accelerator.is_main_process:
        try:
            import wandb
            wandb_config = get_wandb_config()
            wandb_config["DEVICE"] = str(accelerator.device) 
            wandb.init(project=WANDB_PROJECT, group=WANDB_GROUP, tags=WANDB_TAGS, config=wandb_config)
        except ImportError:
            logging.warning("wandb not installed, skipping wandb logging.")
            WANDB_LOG_TRAIN = False 
        
    if accelerator.is_main_process:
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(SAMPLE_DIR, exist_ok=True)
    
    set_seed(SEED)

    logging.info(f"Using device: {accelerator.device}")
    logging.info(f"Effective batch size: {EFFECTIVE_BATCH_SIZE} (Per-device: {BATCH_SIZE} * Grad Accum: {GRADIENT_ACCUMULATION_STEPS})")
    logging.info(f"Max grad norm for clipping: {MAX_GRAD_NORM}")
    logging.info(f"Self-conditioning globally enabled: {SELF_CONDITION}")
    logging.info(f"Model predicting x0: {PRED_X0}")
    logging.info(f"P2 Loss Weighting: k={P2_LOSS_WEIGHT_K}, gamma={P2_LOSS_WEIGHT_GAMMA}")

    dataloader = get_dataset(
        data_path=DATA_PATH, embedding_path=EMBEDDING_PATH, batch_size=BATCH_SIZE,
        voxel_shape=VOXEL_SHAPE, embedding_dim=EMBEDDING_DIM,
        standardize_embeddings=STANDARDIZE_EMBEDDINGS, num_workers=0, shuffle=True
    )

    model, optimizer = create_model_and_optimizer()
    if accelerator.is_main_process: 
        logging.info(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    unet_in_channels_ema_main_path = EMBEDDING_DIM * 2 if SELF_CONDITION else EMBEDDING_DIM
    ema_model = UNet3D(
        dim=MODEL_DIM, dim_mults=DIM_MULTS, in_channels=unet_in_channels_ema_main_path,
        out_dim=UNET_OUT_CHANNELS, resnet_block_groups=8, use_linear_attention=True,
        attn_heads=4, attn_dim_head=32
    ).to(accelerator.device) 
    copy_params_to_ema(model, ema_model) 
    ema_model.eval()

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    start_step = 0 
    
    ddpm_params_config = {
        'beta_schedule': BETA_SCHEDULE, 
        'timesteps': TIMESTEPS,
        'p2_loss_weight_gamma': P2_LOSS_WEIGHT_GAMMA, 
        'p2_loss_weight_k': P2_LOSS_WEIGHT_K,         
        'device': accelerator.device 
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
        disable=not accelerator.is_main_process,
        dynamic_ncols=True
    )
    
    global_data_step_counter = 0 

    while completed_optimizer_steps < MAX_STEP:
        model.train() 
        for batch_idx, x_start in enumerate(dataloader):
            t_rand = torch.randint(0, ddpm_params['timesteps'], (x_start.shape[0],), device=accelerator.device).long()

            with accelerator.accumulate(model):
                loss = p_loss(model, x_start, t_rand, ddpm_params,
                              loss_type=LOSS_TYPE, self_condition=SELF_CONDITION,
                              pred_x0=PRED_X0, 
                              p2_loss_weight_gamma=P2_LOSS_WEIGHT_GAMMA)
                
                accelerator.backward(loss)
                
                # Gradient Clipping
                if MAX_GRAD_NORM is not None and MAX_GRAD_NORM > 0: # Only clip if a value is set
                    accelerator.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                
                optimizer.step()
                optimizer.zero_grad()

            global_data_step_counter += 1

            if accelerator.sync_gradients: 
                progress_bar.update(1)
                completed_optimizer_steps += 1

                unwrapped_model = accelerator.unwrap_model(model)
                if completed_optimizer_steps >= EMA_UPDATE_AFTER_STEP and \
                   (completed_optimizer_steps - EMA_UPDATE_AFTER_STEP) % EMA_UPDATE_EVERY == 0: 
                    current_ema_decay = ema_decay_fn(completed_optimizer_steps)
                    apply_ema_decay(unwrapped_model, ema_model, current_ema_decay)

                if completed_optimizer_steps % LOG_EVERY_STEPS == 0:
                    avg_loss_accumulated = loss.item() 
                    log_data = {'step': completed_optimizer_steps, 'loss': avg_loss_accumulated} 
                    if accelerator.is_main_process:
                        # logging.info(f"Step: {completed_optimizer_steps}, Loss: {avg_loss_accumulated:.4f}")
                        if WANDB_LOG_TRAIN:
                            wandb.log(log_data, step=completed_optimizer_steps)
                
                if completed_optimizer_steps > 0 and completed_optimizer_steps % SAVE_AND_SAMPLE_EVERY == 0 :
                    if accelerator.is_main_process:
                        logging.info(f"Saving checkpoint and generating samples at optimizer step {completed_optimizer_steps}...")
                        
                        unwrapped_model_to_save = accelerator.unwrap_model(model)
                        unwrapped_ema_model_to_save = ema_model

                        save_checkpoint(
                            model=unwrapped_model_to_save, optimizer=optimizer, step=completed_optimizer_steps, 
                            checkpoint_dir=CHECKPOINT_DIR, ema_model=unwrapped_ema_model_to_save, 
                            filename_prefix="checkpoint"
                        )
                        
                        unwrapped_ema_model_to_save.eval() 
                        sampler_step_fn = functools.partial(
                            ddpm_sample_step, 
                            model=unwrapped_ema_model_to_save, 
                            ddpm_params=ddpm_params, 
                            pred_x0=PRED_X0,
                            self_condition=SELF_CONDITION
                        )
                        
                        sampler_kwargs = {}
                        if CLIP_DENOISED_VALUE is not None:
                             sampler_kwargs['clip_denoised_value'] = CLIP_DENOISED_VALUE

                        generated_sample_embeddings = sample_loop(
                            shape=(NUM_SAMPLE, UNET_OUT_CHANNELS, *VOXEL_SHAPE),
                            timesteps=ddpm_params['timesteps'], device=accelerator.device,
                            sampler_fn=sampler_step_fn, 
                            progress_desc=f"Sampling at step {completed_optimizer_steps}", 
                            **sampler_kwargs
                        )
                        
                        sample_block_ids_save_path = os.path.join(SAMPLE_DIR, f"sample_block_ids_step_{completed_optimizer_steps}.npy")
                        
                        save_samples_as_block_ids(
                            generated_sample_embeddings=generated_sample_embeddings,
                            embedding_path_for_reference=EMBEDDING_PATH,
                            embedding_dim_for_reference=EMBEDDING_DIM,
                            model_outputs_standardized_embeddings=STANDARDIZE_EMBEDDINGS,
                            output_path_npy=sample_block_ids_save_path,
                            metric_for_id_mapping=SAMPLE_TO_BLOCK_ID_METRIC
                        )
                        
                        if WANDB_LOG_TRAIN:
                            wandb.log({"sample_block_ids_saved_path": sample_block_ids_save_path}, step=completed_optimizer_steps)
                    
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
    train()
