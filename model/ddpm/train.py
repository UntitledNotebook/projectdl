import os
import time
import logging
import functools
from tqdm import tqdm
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from flax import jax_utils
from typing import Any, Dict, Tuple, Optional

from model import UNet3D
from train_state import TrainState
from data import get_dataset
from diffusion import p_loss, ddpm_sample_step, get_ddpm_params, sample_loop
from utils import l2_loss, l1_loss, create_ema_decay_schedule, apply_ema_decay, copy_params_to_ema, save_checkpoint

# Global Configurations
MAX_STEP = 100
DATA_PATH = "data/block_ids_32_32.npy"  # Path to .npy file: (n_samples, 32, 32, 32), np.int32
EMBEDDING_PATH = "output/block2vec/block_embeddings.npy"  # Path to .npy file: (num_blocks, embedding_dim)
OUTPUT_DIR = "output/ddpm"  # Base directory for outputs
CHECKPOINT_DIR = os.path.abspath(os.path.join(OUTPUT_DIR, "checkpoints"))  # Subdir for model checkpoints
SAMPLE_DIR = os.path.abspath(os.path.join(OUTPUT_DIR, "samples"))  # Subdir for generated samples
SEED = 0
BATCH_SIZE = 32
VOXEL_SHAPE = (32, 32, 32)  # (depth, height, width)
EMBEDDING_DIM = 4  # Matches embedding_dim in EMBEDDING_PATH
STANDARDIZE_EMBEDDINGS = True  # Whether to standardize embeddings (zero mean, unit variance)
MODEL_DIM = 32
DIM_MULTS = (1, 2, 4)
BETA_SCHEDULE = 'cosine'
TIMESTEPS = 1000
SELF_CONDITION = False
PRED_X0 = False
P2_LOSS_WEIGHT_GAMMA = 1.0
P2_LOSS_WEIGHT_K = 1.0
HALF_PRECISION = False
LOSS_TYPE = 'l2'
LOG_EVERY_STEPS = 1
SAVE_AND_SAMPLE_EVERY = 10
NUM_SAMPLE = 8
LEARNING_RATE = 1e-4
BETA1 = 0.9
BETA2 = 0.999
EPS = 1e-8
EMA_UPDATE_AFTER_STEP = 5
EMA_UPDATE_EVERY = 1
EMA_INV_GAMMA = 1.0
EMA_POWER = 0.75
EMA_MIN_VALUE = 0.0
EMA_BETA = 0.9999
WANDB_LOG_TRAIN = True
WANDB_PROJECT = 'projectdl'
WANDB_JOB_TYPE = 'train'
WANDB_GROUP = 'ddpm'
WANDB_TAG = ['debug']

def create_model(model_cls, half_precision: bool) -> Any:
    """Creates the UNet3D model with specified precision.

    Args:
        model_cls (Any): Model class (UNet3D).
        half_precision (bool): If True, use half-precision (bfloat16 on TPU, float16 on GPU).

    Returns:
        Any: Instantiated model.
    """
    platform = jax.local_devices()[0].platform
    model_dtype = jnp.bfloat16 if half_precision and platform == 'tpu' else jnp.float16 if half_precision else jnp.float32
    return model_cls(dtype=model_dtype, dim=MODEL_DIM, out_dim=EMBEDDING_DIM, dim_mults=DIM_MULTS)

def initialized(key: jax.random.PRNGKey, voxel_shape: Tuple[int, ...], model: Any) -> Dict[str, Any]:
    """Initializes model parameters.

    Args:
        key (jax.random.PRNGKey): Random key for initialization.
        voxel_shape (Tuple[int, ...]): Shape of voxel grid including embedding dimension.
        model (Any): Model instance.

    Returns:
        Dict[str, Any]: Initialized parameters.
    """
    input_shape = (1, *voxel_shape)
    @jax.jit
    def init(*args):
        return model.init(*args)
    variables = init(
        {'params': key},
        jnp.ones(input_shape, model.dtype),
        jnp.ones(input_shape[:1], model.dtype)
    )
    return variables['params']

def create_train_state(rng: jax.random.PRNGKey) -> TrainState:
    """Creates the initial training state.

    Args:
        rng (jax.random.PRNGKey): Random key for initialization.

    Returns:
        TrainState: Initial training state.
    """
    from flax.training import dynamic_scale as dynamic_scale_lib
    dynamic_scale = dynamic_scale_lib.DynamicScale() if HALF_PRECISION and jax.local_devices()[0].platform == 'gpu' else None
    model = create_model(model_cls=UNet3D, half_precision=HALF_PRECISION)
    rng, rng_params = jax.random.split(rng)
    params = initialized(rng_params, (*VOXEL_SHAPE, EMBEDDING_DIM), model)
    tx = optax.adam(learning_rate=LEARNING_RATE, b1=BETA1, b2=BETA2, eps=EPS)
    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        params_ema=params,
        dynamic_scale=dynamic_scale
    )

def train():
    """Trains the diffusion model for Minecraft landscape generation."""
    if WANDB_LOG_TRAIN and jax.process_index() == 0:
        wandb_config = {
            'seed': SEED,
            'data': {
                'batch_size': BATCH_SIZE,
                'voxel_shape': VOXEL_SHAPE,
                'embedding_dim': EMBEDDING_DIM,
                'standardize_embeddings': STANDARDIZE_EMBEDDINGS,
            },
            'model': {
                'dim': MODEL_DIM,
                'dim_mults': DIM_MULTS
            },
            'ddpm': {
                'beta_schedule': BETA_SCHEDULE,
                'timesteps': TIMESTEPS,
                'self_condition': SELF_CONDITION,
                'pred_x0': PRED_X0,
                'p2_loss_weight_gamma': P2_LOSS_WEIGHT_GAMMA,
                'p2_loss_weight_k': P2_LOSS_WEIGHT_K
            },
            'training': {
                'half_precision': HALF_PRECISION,
                'num_train_steps': MAX_STEP,
                'log_every_steps': LOG_EVERY_STEPS,
                'save_and_sample_every': SAVE_AND_SAMPLE_EVERY,
                'num_sample': NUM_SAMPLE,
                'loss_type': LOSS_TYPE
            },
            'optim': {
                'lr': LEARNING_RATE,
                'beta1': BETA1,
                'beta2': BETA2,
                'eps': EPS
            },
            'ema': {
                'update_after_step': EMA_UPDATE_AFTER_STEP,
                'update_every': EMA_UPDATE_EVERY,
                'inv_gamma': EMA_INV_GAMMA,
                'power': EMA_POWER,
                'min_value': EMA_MIN_VALUE,
                'beta': EMA_BETA
            }
        }
        wandb.init(
            project=WANDB_PROJECT,
            group=WANDB_GROUP,
            job_type=WANDB_JOB_TYPE,
            tags=WANDB_TAG,
            config=wandb_config
        )
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    rng = jax.random.PRNGKey(SEED)
    rng, d_rng = jax.random.split(rng)
    get_batch = get_dataset(d_rng, DATA_PATH, EMBEDDING_PATH, BATCH_SIZE, VOXEL_SHAPE, EMBEDDING_DIM, STANDARDIZE_EMBEDDINGS)
    rng, state_rng = jax.random.split(rng)
    state = create_train_state(state_rng)
    from flax.training import checkpoints
    state = checkpoints.restore_checkpoint(CHECKPOINT_DIR, state)
    if jax.process_index() == 0:
        unreplicated_params = jax_utils.unreplicate(state.params)
        param_count = sum(x.size for x in jax.tree_util.tree_leaves(unreplicated_params))
        logging.info(f"Total number of parameters: {param_count}")
        if WANDB_LOG_TRAIN:
            wandb.summary['total_parameters'] = param_count
    step_offset = int(state.step)
    state = jax_utils.replicate(state)
    loss_fn = l2_loss if LOSS_TYPE == 'l2' else l1_loss
    ddpm_params = get_ddpm_params({
        'beta_schedule': BETA_SCHEDULE,
        'timesteps': TIMESTEPS,
        'p2_loss_weight_gamma': P2_LOSS_WEIGHT_GAMMA,
        'p2_loss_weight_k': P2_LOSS_WEIGHT_K
    })
    ema_decay_fn = create_ema_decay_schedule({
        'update_after_step': EMA_UPDATE_AFTER_STEP,
        'update_every': EMA_UPDATE_EVERY,
        'inv_gamma': EMA_INV_GAMMA,
        'power': EMA_POWER,
        'min_value': EMA_MIN_VALUE,
        'beta': EMA_BETA
    })
    train_step = functools.partial(
        p_loss, ddpm_params=ddpm_params, loss_fn=loss_fn,
        self_condition=SELF_CONDITION, is_pred_x0=PRED_X0, pmap_axis='batch'
    )
    p_train_step = jax.pmap(train_step, axis_name='batch')
    p_apply_ema = jax.pmap(apply_ema_decay, in_axes=(0, None), axis_name='batch')
    p_copy_params_to_ema = jax.pmap(copy_params_to_ema, axis_name='batch')
    sample_step = functools.partial(
        ddpm_sample_step, ddpm_params=ddpm_params,
        self_condition=SELF_CONDITION, is_pred_x0=PRED_X0
    )
    p_sample_step = jax.pmap(sample_step, axis_name='batch')
    train_metrics = []
    train_metrics_last_t = time.time()
    logging.info('Initial compilation, this might take some minutes...')
    for step in tqdm(range(step_offset, MAX_STEP)):
        rng, *train_step_rng = jax.random.split(rng, num=jax.local_device_count() + 1)
        train_step_rng = jnp.asarray(train_step_rng)
        batch = get_batch()
        state, metrics = p_train_step(train_step_rng, state, batch)
        if step == step_offset:
            logging.info('Initial compilation completed.')
            logging.info(f"Number of devices: {batch['voxel'].shape[0]}")
            logging.info(f"Batch size per device: {batch['voxel'].shape[1]}")
            logging.info(f"Input shape: {batch['voxel'].shape[2:]}")
        if (step + 1) <= EMA_UPDATE_AFTER_STEP:
            state = p_copy_params_to_ema(state)
        elif (step + 1) % EMA_UPDATE_EVERY == 0:
            ema_decay = ema_decay_fn(step)
            logging.info(f'Update EMA parameters with decay rate {ema_decay}')
            state = p_apply_ema(state, ema_decay)
        if LOG_EVERY_STEPS:
            train_metrics.append(metrics)
            if (step + 1) % LOG_EVERY_STEPS == 0:
                summary = {}
                for k in train_metrics[0].keys():
                    vals = np.array([m[k] for m in train_metrics])
                    summary[f'train/{k}'] = float(vals.mean())
                summary['time/seconds_per_step'] = (time.time() - train_metrics_last_t) / LOG_EVERY_STEPS
                train_metrics = []
                train_metrics_last_t = time.time()
                if WANDB_LOG_TRAIN:
                    wandb.log({"train/step": step + 1, **summary})
        if (step + 1) % SAVE_AND_SAMPLE_EVERY == 0 or step + 1 == MAX_STEP:
            logging.info('Generating samples...')
            samples = []
            for i in range(0, NUM_SAMPLE, BATCH_SIZE):
                rng, sample_rng = jax.random.split(rng)
                samples.append(sample_loop(
                    sample_rng, state,
                    (jax.local_device_count(), BATCH_SIZE // jax.local_device_count(), *VOXEL_SHAPE, EMBEDDING_DIM),
                    p_sample_step, TIMESTEPS
                ))
            samples = jnp.concatenate(samples)
            this_sample_dir = os.path.join(SAMPLE_DIR, f"iter_{step}_host_{jax.process_index()}")
            os.makedirs(this_sample_dir, exist_ok=True)
            samples_array = np.array(samples)
            np.save(os.path.join(this_sample_dir, "sample.npy"), samples_array)
            save_checkpoint(state, CHECKPOINT_DIR)
    jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()
    return state