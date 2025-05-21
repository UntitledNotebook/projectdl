import jax
import jax.numpy as jnp
import jax.nn
import numpy as np
import json
from pathlib import Path
import random
from functools import partial
import optax
from optax import cosine_decay_schedule # Add this import
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import multiprocessing
import os
import itertools
import time
from openTSNE import TSNE

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(processName)s - %(message)s')
logger = logging.getLogger(__name__)

# Set random seed for reproducibility (for main process JAX ops and initial setup)
# Worker processes will be seeded separately.
random.seed(42)
np.random.seed(42)
rng = jax.random.PRNGKey(42) # Initial JAX PRNG key for the main process

# Configuration
DATA_DIR = Path("data")
OUTPUT_DIR = Path(f"output/block2vec/{time.strftime('%Y%m%d_%H%M%S')}")
NPY_FILE = DATA_DIR / "block_ids_32_32.npy"
EMBEDDING_DIM = 8
MAX_WINDOW_SIZE = 11
NEGATIVE_SAMPLES = 5
SUBSAMPLE_THRESHOLD = 1e-5
LEARNING_RATE = 0.020
BATCH_SIZE = 1024
MAX_STEPS = 5000
WANDB_PROJECT = "projectdl"
WANDB_GROUP = "block2vec"
WANDB_TAG = ["cosine"] # Added mmap tag
CHECKPOINT_INTERVAL = 1000
TSNE_SUBSAMPLE = 30
NUM_WORKERS = max(1, os.cpu_count() - 2)

if not OUTPUT_DIR.exists():
    OUTPUT_DIR.mkdir(parents=True)

# --- Worker global variables ---
g_block_ids = None
g_subsample_probs = None
g_log_uniform_probs = None
g_vocab_size = None
g_max_window_size = None
g_negative_samples_const = None
g_x_len, g_y_len, g_z_len = None, None, None
g_n_samples = None
g_rel_coords_dict = None
# --- End Worker global variables ---

def worker_init_fn(block_ids_data, subsample_probs_data, log_uniform_probs_data,
                   vocab_size_data, max_window_size_data, negative_samples_data,
                   block_ids_shape_data, n_samples_data):
    global g_block_ids, g_subsample_probs, g_log_uniform_probs, g_vocab_size
    global g_max_window_size, g_negative_samples_const, g_x_len, g_y_len, g_z_len, g_n_samples
    global g_rel_coords_dict

    worker_pid = os.getpid()
    logger.debug(f"Initializing worker PID: {worker_pid}.")
    g_block_ids = block_ids_data
    g_subsample_probs = subsample_probs_data
    g_log_uniform_probs = log_uniform_probs_data
    g_vocab_size = vocab_size_data
    g_max_window_size = max_window_size_data
    g_negative_samples_const = negative_samples_data
    g_n_samples = n_samples_data
    g_x_len, g_y_len, g_z_len = block_ids_shape_data[1], block_ids_shape_data[2], block_ids_shape_data[3]

    # Precompute rel_coords for each window size
    g_rel_coords_dict = {}
    for w in range(1, g_max_window_size + 1):
        axis_range = np.arange(-w, w + 1, dtype=np.int32)
        dx_grid, dy_grid, dz_grid = np.meshgrid(axis_range, axis_range, axis_range, indexing='ij')
        rel_coords = np.stack([dx_grid.ravel(), dy_grid.ravel(), dz_grid.ravel()], axis=-1)
        is_not_zero_offset = np.any(rel_coords != 0, axis=1)
        g_rel_coords_dict[w] = rel_coords[is_not_zero_offset]

    process_seed = worker_pid + int(time.time() * 10000) % 1000000
    random.seed(process_seed)
    np.random.seed(process_seed)
    logger.debug(f"Worker PID: {worker_pid} seeded with {process_seed}.")

def generate_single_sample_worker(_=None):
    while True:
        sample_idx = random.randint(0, g_n_samples - 1)
        x = random.randint(0, g_x_len - 1)
        y = random.randint(0, g_y_len - 1)
        z = random.randint(0, g_z_len - 1)

        chunk = g_block_ids[sample_idx]
        target_id = int(chunk[x, y, z])

        if random.random() < g_subsample_probs.get(target_id, 0):
            continue

        current_window_size = random.randint(1, g_max_window_size)
        rel_coords = g_rel_coords_dict[current_window_size]

        abs_coords_x = x + rel_coords[:, 0]
        abs_coords_y = y + rel_coords[:, 1]
        abs_coords_z = z + rel_coords[:, 2]

        valid_mask = (
            (abs_coords_x >= 0) & (abs_coords_x < g_x_len) &
            (abs_coords_y >= 0) & (abs_coords_y < g_y_len) &
            (abs_coords_z >= 0) & (abs_coords_z < g_z_len)
        )

        valid_rel_coords = rel_coords[valid_mask]
        if valid_rel_coords.shape[0] == 0:
            continue

        final_abs_coords_x = abs_coords_x[valid_mask]
        final_abs_coords_y = abs_coords_y[valid_mask]
        final_abs_coords_z = abs_coords_z[valid_mask]

        distances = np.abs(valid_rel_coords[:, 0]) + np.abs(valid_rel_coords[:, 1]) + np.abs(valid_rel_coords[:, 2])
        distances = np.maximum(1, distances)
        inv_dist_probs = 1.0 / distances
        
        random_draws = np.random.rand(len(inv_dist_probs))
        selected_mask = random_draws < inv_dist_probs
        
        selected_coords_x = final_abs_coords_x[selected_mask]
        selected_coords_y = final_abs_coords_y[selected_mask]
        selected_coords_z = final_abs_coords_z[selected_mask]

        if selected_coords_x.shape[0] == 0:
            continue

        ctx_ids_np = chunk[selected_coords_x, selected_coords_y, selected_coords_z]
        context_ids_list = [int(cid) for cid in ctx_ids_np]
        if not context_ids_list:
            continue
        context_id = random.choice(context_ids_list)

        neg_context_ids = np.random.choice(
            g_vocab_size, size=g_negative_samples_const, p=g_log_uniform_probs, replace=True
        )
        return target_id, context_id, neg_context_ids.astype(np.int32)


def load_data():
    logger.info(f"Loading block IDs from {NPY_FILE} using memory mapping (mmap_mode='r').")
    try:
        # Use mmap_mode='r' for read-only memory mapping
        block_ids = np.load(NPY_FILE, mmap_mode='r')
        logger.info(f"Successfully memory-mapped block IDs. Shape: {block_ids.shape}, Dtype: {block_ids.dtype}")
    except FileNotFoundError:
        logger.error(f"Error: The NPY file {NPY_FILE} was not found.")
        raise
    except ValueError as e:
        logger.error(f"Error loading NPY file {NPY_FILE}. It might be corrupted or not a valid NumPy file: {e}")
        raise
    
    logger.info("Loading ID-to-SNBT mapping...")
    with open(DATA_DIR / "id_to_snbt.json", "r") as f:
        id_to_snbt = json.load(f)
        id_to_snbt = {int(k): v for k, v in id_to_snbt.items()}
    
    logger.info("Loading SNBT counts...")
    with open(DATA_DIR / "snbt_counts.json", "r") as f:
        snbt_counts = json.load(f)
    
    block_counts = {}
    # Ensure all items in id_to_snbt have a count, default to 0 if missing
    for bid, snbt in id_to_snbt.items():
        block_counts[bid] = snbt_counts.get(snbt, 0)
        if snbt not in snbt_counts:
             logger.warning(f"SNBT '{snbt}' (ID: {bid}) not found in snbt_counts.json. Assigning count 0.")

    total_count = sum(block_counts.values())
    if total_count == 0:
        logger.error("Total block count is zero. Cannot compute probabilities. Check snbt_counts.json.")
        raise ValueError("Total block count is zero. SNBT counts might be missing or all zero.")
        
    # vocab_size should be the number of unique block IDs, typically max_id + 1
    # Assuming IDs in id_to_snbt are dense from 0 to N-1
    vocab_size = len(id_to_snbt) if id_to_snbt else 0
    if not id_to_snbt: # If id_to_snbt is empty
        actual_max_id = np.max(block_ids) if block_ids.size > 0 else -1
        vocab_size = actual_max_id + 1
        logger.warning(f"id_to_snbt.json is empty or not loaded correctly. Inferring vocab_size as {vocab_size} from block_ids max value. SNBT names will be unavailable.")
        # Create a dummy id_to_snbt if it's critical for other parts, e.g., visualization
        id_to_snbt = {i: f"minecraft:unknown_block_id_{i}" for i in range(vocab_size)}


    block_probs = {bid: count / total_count for bid, count in block_counts.items()}
    subsample_probs = {
        bid: 1 - np.sqrt(SUBSAMPLE_THRESHOLD / (block_probs.get(bid, 0) + 1e-10))
        for bid in range(vocab_size) # Ensure all possible IDs have a subsample prob
    }
    
    all_block_counts_for_ranking = np.zeros(vocab_size, dtype=float) # Use float for counts to avoid overflow if counts are huge
    for bid_int in range(vocab_size):
        all_block_counts_for_ranking[bid_int] = block_counts.get(bid_int, 0)
            
    sorted_block_indices = np.argsort(-all_block_counts_for_ranking)
    block_ranks = {int(bid): rank + 1 for rank, bid in enumerate(sorted_block_indices)}
    
    log_uniform_probs = np.zeros(vocab_size, dtype=float)
    for bid_int in range(vocab_size):
        rank = block_ranks.get(bid_int)
        if rank is not None and rank > 0:
             log_uniform_probs[bid_int] = np.log( (rank + 1) / rank ) # More stable version of log(1 + 1/rank)
        else: # Handles blocks with 0 count or if bid_int somehow not in ranks
            log_uniform_probs[bid_int] = np.log((vocab_size +1.0) / vocab_size) # Smallest possible probability if using this scheme
            logger.debug(f"Block ID {bid_int} has rank {rank}, assigning low probability.")

    sum_log_uniform_probs = np.sum(log_uniform_probs)
    if sum_log_uniform_probs == 0 or not np.isfinite(sum_log_uniform_probs):
        logger.warning(f"Sum of log_uniform_probs is {sum_log_uniform_probs}. Defaulting to uniform distribution for negative sampling.")
        log_uniform_probs = np.ones(vocab_size) / vocab_size
    else:
        log_uniform_probs /= sum_log_uniform_probs

    if np.any(np.isnan(log_uniform_probs)) or np.any(np.isinf(log_uniform_probs)):
        logger.error("NaN or Inf found in log_uniform_probs after normalization. Defaulting to uniform.")
        log_uniform_probs = np.ones(vocab_size) / vocab_size
    if not (np.abs(np.sum(log_uniform_probs) - 1.0) < 1e-5): # Check if sums to 1
        logger.warning(f"log_uniform_probs does not sum to 1 (sum={np.sum(log_uniform_probs)}). Re-normalizing.")
        log_uniform_probs /= np.sum(log_uniform_probs)


    logger.info("Data loading and preprocessing complete.")
    return block_ids, id_to_snbt, block_counts, subsample_probs, vocab_size, log_uniform_probs


def init_model(vocab_size, embedding_dim, key):
    key, target_embed_key = jax.random.split(key)
    key, context_embed_key = jax.random.split(key)
    target_embeddings = jax.random.uniform(target_embed_key, (vocab_size, embedding_dim), minval=-0.01, maxval=0.01)
    context_embeddings = jax.random.uniform(context_embed_key, (vocab_size, embedding_dim), minval=-0.01, maxval=0.01)
    return {"target": target_embeddings, "context": context_embeddings}, key

@partial(jax.jit)
def nce_loss(params, target_ids, context_ids, neg_context_ids):
    target_embed = params["target"][target_ids]
    context_embed = params["context"][context_ids]
    neg_context_embed = params["context"][neg_context_ids]
    
    pos_logits = jnp.sum(target_embed * context_embed, axis=1)
    # Using log_sigmoid for numerical stability: log(sigmoid(x)) = -softplus(-x)
    pos_loss = -jax.nn.log_sigmoid(pos_logits)
    
    neg_logits = jnp.sum(target_embed[:, None, :] * neg_context_embed, axis=2)
    # log(sigmoid(-x)) = -softplus(x)
    neg_loss = -jnp.sum(jax.nn.log_sigmoid(-neg_logits), axis=1)
    
    return jnp.mean(pos_loss + neg_loss)

@partial(jax.jit, static_argnums=(5,))
def update_step(params, opt_state, target_ids, context_ids, neg_context_ids, optimizer):
    loss, grads = jax.value_and_grad(nce_loss)(params, target_ids, context_ids, neg_context_ids)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

def visualize_embeddings(embeddings_np, id_to_snbt_map, block_counts_map, vocab_s, step_num=None):
    # Ensure embeddings_np is a NumPy array on CPU for t-SNE
    embeddings_np = np.asarray(embeddings_np)

    if TSNE_SUBSAMPLE is not None and vocab_s > TSNE_SUBSAMPLE:
        valid_indices_counts = {bid: count for bid, count in block_counts_map.items() if 0 <= bid < embeddings_np.shape[0]}
        top_blocks = sorted(valid_indices_counts.items(), key=lambda x: x[1], reverse=True)[:TSNE_SUBSAMPLE]
        indices = [bid for bid, _ in top_blocks]
        if not indices:
            logger.warning("TSNE: No valid blocks for subsampling. Visualizing all available embeddings.")
            sub_embeddings = embeddings_np
            indices = list(range(embeddings_np.shape[0]))
        else:
            sub_embeddings = embeddings_np[np.array(indices)]
    else:
        sub_embeddings = embeddings_np
        indices = list(range(embeddings_np.shape[0]))

    # Filter out any potential all-zero embeddings if they cause issues, though t-SNE should handle them.
    non_zero_rows_mask = np.any(sub_embeddings != 0, axis=1)
    if not np.all(non_zero_rows_mask) and np.sum(non_zero_rows_mask) > 2: # if some are all zero
        logger.info(f"TSNE: Found {np.sum(~non_zero_rows_mask)} all-zero embedding vectors. Visualizing non-zero ones.")
        sub_embeddings_filtered = sub_embeddings[non_zero_rows_mask]
        indices_filtered = [idx for i, idx in enumerate(indices) if non_zero_rows_mask[i]]
    else:
        sub_embeddings_filtered = sub_embeddings
        indices_filtered = indices
        
    if sub_embeddings_filtered.shape[0] < 2:
        logger.warning(f"TSNE: Not enough samples ({sub_embeddings_filtered.shape[0]}) for visualization at step {step_num}.")
        return

    # Ensure perplexity is less than the number of samples
    perplexity_val = min(8, sub_embeddings_filtered.shape[0] - 1)
    if perplexity_val <=0: # if only 1 sample after filtering
        logger.warning(f"TSNE: Perplexity value ({perplexity_val}) is too low. Skipping t-SNE.")
        return

    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_val, n_iter=1000, n_jobs=1) # use all cores for t-SNE
    embeddings_2d = tsne.fit(sub_embeddings_filtered)
    
    plt.figure(figsize=(12, 10))
    sns.scatterplot(x=embeddings_2d[:, 0], y=embeddings_2d[:, 1], alpha=0.7, s=50)
    
    num_labels = min(50, embeddings_2d.shape[0])
    for i in range(num_labels):
        original_block_idx = indices_filtered[i]
        block_name_full = id_to_snbt_map.get(original_block_idx, f"ID_{original_block_idx}")
        block_name = block_name_full.split(':')[-1].split('[')[0]
        plt.text(embeddings_2d[i, 0], embeddings_2d[i, 1], block_name, fontsize=7, alpha=0.8)
    
    title = f"t-SNE Projection (Step {step_num})" if step_num is not None else "t-SNE Projection (Final)"
    plt.title(title, fontsize=14)
    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    base_filename = f"tsne_step_{step_num}" if step_num is not None else "tsne_final"
    svg_path = OUTPUT_DIR / f"{base_filename}.svg"
    png_path = OUTPUT_DIR / f"{base_filename}.png"
    
    plt.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.savefig(png_path, format="png", bbox_inches="tight", dpi=300)
    plt.close()
    
    try:
        wandb.log({f"tsne_visualization_{'' if step_num is not None else 'final'}": wandb.Image(str(png_path))})
    except Exception as e:
        logger.error(f"Failed to log t-SNE image to wandb: {e}")


def train_model(block_ids_data, subsample_probs_data, log_uniform_probs_data,
                vocab_s, embedding_d, max_s, id_to_snbt_map, block_counts_map, initial_jax_rng):
    
    run = wandb.init(
        project=WANDB_PROJECT,
        group=WANDB_GROUP,
        config={
            "embedding_dim": embedding_d,
            "max_window_size": MAX_WINDOW_SIZE,
            "negative_samples": NEGATIVE_SAMPLES,
            "subsample_threshold": SUBSAMPLE_THRESHOLD,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "max_steps": max_s,
            "num_workers": NUM_WORKERS,
        },
        tags=WANDB_TAG,
    )
    
    current_jax_rng, model_init_rng = jax.random.split(initial_jax_rng)
    params, current_jax_rng = init_model(vocab_s, embedding_d, model_init_rng)
    
    # Define cosine learning rate schedule
    lr_schedule = cosine_decay_schedule(
        init_value=LEARNING_RATE,
        decay_steps=max_s 
    )
    
    optimizer = optax.adam(learning_rate=lr_schedule) # Use the schedule
    opt_state = optimizer.init(params)
    
    n_samples_data = block_ids_data.shape[0]
    block_ids_shape_data = block_ids_data.shape

    pool_init_args = (
        block_ids_data, subsample_probs_data, log_uniform_probs_data,
        vocab_s, MAX_WINDOW_SIZE, NEGATIVE_SAMPLES,
        block_ids_shape_data, n_samples_data
    )
    
    logger.info(f"Starting training with {NUM_WORKERS} data generation workers.")
    chunk_size = max(1, BATCH_SIZE // (NUM_WORKERS * 4 if NUM_WORKERS > 0 else 1))
    logger.info(f"Using chunksize {chunk_size} for multiprocessing pool tasks.")

    # Using 'spawn' context for broader compatibility, though 'fork' is more memory efficient for mmap with workers.
    # On Linux, 'fork' is default and preferred. This is just for explicit demonstration or if issues arise.
    # mp_context = multiprocessing.get_context('spawn') # Or 'fork' or 'forkserver'
    # with mp_context.Pool(processes=NUM_WORKERS, initializer=worker_init_fn, initargs=pool_init_args) as pool:
    
    # Default context is usually fine. 'fork' on Linux/macOS, 'spawn' on Windows.
    with multiprocessing.Pool(processes=NUM_WORKERS, initializer=worker_init_fn, initargs=pool_init_args) as pool:
        sample_generator_iterator = pool.imap_unordered(
            generate_single_sample_worker, 
            itertools.count(),
            chunksize=chunk_size
        )

        pbar = tqdm(total=max_s, desc="Training Steps")
        for step in range(max_s):
            batch_target_ids = []
            batch_context_ids = []
            batch_neg_context_ids = []

            try:
                for _ in range(BATCH_SIZE):
                    target_id, context_id, neg_ids = next(sample_generator_iterator)
                    batch_target_ids.append(target_id)
                    batch_context_ids.append(context_id)
                    batch_neg_context_ids.append(neg_ids)
            except StopIteration:
                logger.error("Sample generator iterator stopped unexpectedly. Ending training.")
                break 
            except Exception as e:
                logger.error(f"Error fetching data from workers at step {step}: {e}", exc_info=True)
                # Potentially try to recover or break
                if step > 0 and len(batch_target_ids) < BATCH_SIZE // 2: # If severely underfilled batch after an error
                    logger.warning(f"Step {step}: Batch significantly underfilled after error. Breaking training.")
                    break
                elif not batch_target_ids:
                     logger.error(f"Step {step}: No data collected after error. Breaking training.")
                     break


            if not batch_target_ids: # If batch is empty (e.g., due to early break from data collection)
                logger.error(f"Step {step}: No data collected for this batch. Skipping update and potentially ending training.")
                pbar.update(1) # Still update pbar for the attempted step
                if step > max_s // 2 and len(batch_target_ids) == 0 : # if consistently failing
                     logger.error("Consistently failing to get data. Aborting training.")
                     break
                continue


            target_ids_jnp = jnp.array(batch_target_ids, dtype=jnp.int32)
            context_ids_jnp = jnp.array(batch_context_ids, dtype=jnp.int32)
            neg_context_ids_jnp = jnp.array(batch_neg_context_ids, dtype=jnp.int32)
            
            params, opt_state, loss = update_step(
                params, opt_state, target_ids_jnp, context_ids_jnp, neg_context_ids_jnp, optimizer
            )
            
            if run: # Check if wandb run is active
                run.log({"step": step, "loss": loss.item(), "learning_rate": LEARNING_RATE}) # Log LR if it changes
            pbar.set_postfix({"loss": loss.item()})
            
            if step > 0 and step % CHECKPOINT_INTERVAL == 0:
                logger.info(f"Step {step}: Saving checkpoint and visualizing embeddings.")
                checkpoint_path = OUTPUT_DIR / f"block_embeddings_step_{step}.npy"
                np.save(checkpoint_path, np.asarray(params["target"]))
                logger.info(f"Saved checkpoint to {checkpoint_path}")
                if run:
                    run.save(str(checkpoint_path)) # Save checkpoint to wandb
                visualize_embeddings(params["target"], id_to_snbt_map, block_counts_map, vocab_s, step_num=step)
            
            pbar.update(1)
        
        pbar.close()

    logger.info("Training loop finished.")
    
    final_embeddings_path = OUTPUT_DIR / "block_embeddings_final.npy"
    logger.info(f"Saving final block embeddings to {final_embeddings_path}")
    np.save(final_embeddings_path, np.asarray(params["target"]))
    if run:
        run.save(str(final_embeddings_path)) # Save final embeddings to wandb
        visualize_embeddings(params["target"], id_to_snbt_map, block_counts_map, vocab_s, step_num=max_s)
        run.finish()
        logger.info("WandB logging finished.")
    else:
        logger.warning("WandB run was not initialized. Skipping final logging.")
        
    return params["target"]

def main():
    # Access the global JAX PRNG key
    global rng
    logger.info("Starting Block2Vec training process.")
    
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info(f"Number of CPU cores available: {os.cpu_count()}, using {NUM_WORKERS} for data generation.")

    try:
        block_ids, id_to_snbt, block_counts, subsample_probs, vocab_size, log_uniform_probs = load_data()
    except Exception as e:
        logger.critical(f"Failed to load data: {e}", exc_info=True)
        return # Exit if data loading fails
    
    logger.info(f"Vocabulary size: {vocab_size}")
    logger.info(f"Max steps: {MAX_STEPS}, Batch size: {BATCH_SIZE}, Embedding dim: {EMBEDDING_DIM}")

    # Pass the initial JAX PRNG key to train_model
    embeddings = train_model(
        block_ids, subsample_probs, log_uniform_probs,
        vocab_size, EMBEDDING_DIM, MAX_STEPS, 
        id_to_snbt, block_counts, rng
    )
    logger.info("Training complete! Final embeddings shape: %s", embeddings.shape)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()