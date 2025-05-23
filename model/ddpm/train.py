import torch
import torch.optim as optim
import numpy as np
import os
import logging
import functools # For functools.partial
from tqdm import tqdm
from typing import Any, Dict, Tuple, Optional

from .model import UNet3D
from .data import get_dataset 
from .diffusion import p_loss, get_ddpm_params, sample_loop, ddpm_sample_step
from .utils import create_ema_decay_schedule, apply_ema_decay, copy_params_to_ema, save_checkpoint

# --- Global Configurations ---
MAX_STEP = 100000
DATA_PATH = "data/block_ids_32_32.npy"
EMBEDDING_PATH = "output/block2vec/block_embeddings.npy"
OUTPUT_DIR = "output/ddpm_minecraft_v4" # Incremented version
CHECKPOINT_DIR = os.path.abspath(os.path.join(OUTPUT_DIR, "checkpoints"))
SAMPLE_DIR = os.path.abspath(os.path.join(OUTPUT_DIR, "samples"))

SEED = 42
BATCH_SIZE = 8 
VOXEL_SHAPE = (32, 32, 32)
EMBEDDING_DIM = 64 
STANDARDIZE_EMBEDDINGS = True

# Model Config
MODEL_DIM = 64
DIM_MULTS = (1, 2, 4, 8)
UNET_OUT_CHANNELS = EMBEDDING_DIM

# Diffusion Config
BETA_SCHEDULE = 'cosine'
TIMESTEPS = 1000
SELF_CONDITION = True # This will be baked into the sampler_fn
PRED_X0 = True 
CLIP_DENOISED_VALUE = 1.0 if STANDARDIZE_EMBEDDINGS else None

# Loss Config
LOSS_TYPE = 'l1'
P2_LOSS_WEIGHT_GAMMA = 0.0

# Optimizer Config
LEARNING_RATE = 1e-4
BETA1 = 0.9
BETA2 = 0.99
EPS = 1e-8

# EMA Config
EMA_UPDATE_AFTER_STEP = 100
EMA_UPDATE_EVERY = 10
EMA_INV_GAMMA = 1.0
EMA_POWER = 0.75
EMA_MIN_VALUE = 0.0
EMA_BETA = 0.9999

# Logging & Saving Config
LOG_EVERY_STEPS = 100
SAVE_AND_SAMPLE_EVERY = 5000
NUM_SAMPLE = 4

# WANDB (Optional)
WANDB_LOG_TRAIN = False
WANDB_PROJECT = 'projectdl'
WANDB_GROUP = 'ddpm'
WANDB_TAGS = ['pytorch', 'debug']
if SELF_CONDITION: WANDB_TAGS.append('self_cond')
if PRED_X0: WANDB_TAGS.append('pred_x0')


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_model_and_optimizer(
    device: torch.device
) -> Tuple[torch.nn.Module, torch.optim.Optimizer]:
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
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2), eps=EPS)
    return model, optimizer

def train():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    if WANDB_LOG_TRAIN:
        try:
            import wandb
            wandb.init(project=WANDB_PROJECT, group=WANDB_GROUP, tags=WANDB_TAGS, config=globals())
        except ImportError:
            logging.warning("wandb not installed, skipping wandb logging.")
            global WANDB_LOG_TRAIN
            WANDB_LOG_TRAIN = False
        
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if DEVICE.type == 'cuda':
        torch.cuda.manual_seed_all(SEED)

    logging.info(f"Using device: {DEVICE}")
    logging.info(f"Self-conditioning globally enabled for training/sampling: {SELF_CONDITION}")
    logging.info(f"Model predicting x0: {PRED_X0}")

    dataloader = get_dataset(
        data_path=DATA_PATH, embedding_path=EMBEDDING_PATH, batch_size=BATCH_SIZE,
        voxel_shape=VOXEL_SHAPE, embedding_dim=EMBEDDING_DIM,
        standardize_embeddings=STANDARDIZE_EMBEDDINGS, num_workers=0, shuffle=True
    )

    model, optimizer = create_model_and_optimizer(DEVICE)
    
    # EMA model should also have its in_channels set according to SELF_CONDITION
    unet_in_channels_ema_main_path = EMBEDDING_DIM * 2 if SELF_CONDITION else EMBEDDING_DIM
    ema_model = UNet3D(
        dim=MODEL_DIM, dim_mults=DIM_MULTS, in_channels=unet_in_channels_ema_main_path,
        out_dim=UNET_OUT_CHANNELS, resnet_block_groups=8, use_linear_attention=True,
        attn_heads=4, attn_dim_head=32
    ).to(DEVICE)
    copy_params_to_ema(model, ema_model)
    ema_model.eval()

    start_step = 0
    latest_checkpoint_path = os.path.join(CHECKPOINT_DIR, "checkpoint_latest.pth")
    if os.path.exists(latest_checkpoint_path):
        logging.info(f"Restoring checkpoint from {latest_checkpoint_path}")
        try:
            checkpoint_data = torch.load(latest_checkpoint_path, map_location=DEVICE)
            model.load_state_dict(checkpoint_data['model_state_dict'])
            optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
            ema_model.load_state_dict(checkpoint_data['ema_model_state_dict'])
            start_step = checkpoint_data['step'] + 1
            logging.info(f"Restored checkpoint. Starting from step {start_step}")
        except Exception as e:
            logging.error(f"Could not load checkpoint: {e}. Starting from scratch.")
            start_step = 0

    ddpm_params_config = {
        'beta_schedule': BETA_SCHEDULE, 'timesteps': TIMESTEPS,
        'p2_loss_weight_gamma': P2_LOSS_WEIGHT_GAMMA, 'device': DEVICE
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
    
    model.train()
    step = start_step
    progress_bar = tqdm(initial=step, total=MAX_STEP, desc="Training Progress", dynamic_ncols=True)
    data_iter = iter(dataloader)
    
    while step < MAX_STEP:
        try:
            x_start = next(data_iter).to(DEVICE) 
        except StopIteration:
            data_iter = iter(dataloader)
            x_start = next(data_iter).to(DEVICE)

        t_rand = torch.randint(0, ddpm_params['timesteps'], (x_start.shape[0],), device=DEVICE).long()

        optimizer.zero_grad(set_to_none=True)
        
        loss = p_loss(model, x_start, t_rand, ddpm_params,
                      loss_type=LOSS_TYPE, self_condition=SELF_CONDITION,
                      pred_x0=PRED_X0, p2_loss_weight_gamma=P2_LOSS_WEIGHT_GAMMA)
        loss.backward()
        optimizer.step()

        if step >= EMA_UPDATE_AFTER_STEP and step % EMA_UPDATE_EVERY == 0:
            current_ema_decay = ema_decay_fn(step)
            apply_ema_decay(model, ema_model, current_ema_decay)

        if step % LOG_EVERY_STEPS == 0:
            log_data = {'step': step, 'loss': loss.item()}
            logging.info(f"Step: {step}, Loss: {loss.item():.4f}")
            if WANDB_LOG_TRAIN:
                wandb.log(log_data, step=step)

        if step > 0 and step % SAVE_AND_SAMPLE_EVERY == 0 :
            logging.info(f"Saving checkpoint and generating samples at step {step}...")
            save_checkpoint(
                model=model, optimizer=optimizer, step=step, checkpoint_dir=CHECKPOINT_DIR,
                ema_model=ema_model, filename_prefix="checkpoint"
            )
            
            ema_model.eval()
            # SELF_CONDITION is now part of the partial function for ddpm_sample_step
            sampler_step_fn = functools.partial(
                ddpm_sample_step, 
                model=ema_model, 
                ddpm_params=ddpm_params, 
                pred_x0=PRED_X0,
                self_condition=SELF_CONDITION # Baked into the sampler_fn
            )
            
            sampler_kwargs = {}
            if CLIP_DENOISED_VALUE is not None:
                 sampler_kwargs['clip_denoised_value'] = CLIP_DENOISED_VALUE

            # sample_loop no longer takes self_condition directly
            samples = sample_loop(
                shape=(NUM_SAMPLE, UNET_OUT_CHANNELS, *VOXEL_SHAPE),
                timesteps=ddpm_params['timesteps'], device=DEVICE,
                sampler_fn=sampler_step_fn, 
                # initial_voxel can be passed if needed
                progress_desc=f"Sampling at step {step}", 
                **sampler_kwargs
            )
            
            sample_save_path = os.path.join(SAMPLE_DIR, f"sample_step_{step}.npy")
            np.save(sample_save_path, samples.cpu().numpy())
            logging.info(f"Saved {NUM_SAMPLE} samples to {sample_save_path}")
            
            if WANDB_LOG_TRAIN:
                wandb.log({"samples_saved_path": sample_save_path}, step=step)
            model.train()
            
        step += 1
        progress_bar.update(1)
        progress_bar.set_postfix(loss=loss.item())

    progress_bar.close()
    logging.info("Training finished.")
    if WANDB_LOG_TRAIN:
        wandb.finish()

if __name__ == '__main__':
    train()
