import jax
import jax.numpy as jnp
import jax.nn  # Add this import
import numpy as np
import json
from pathlib import Path
import random
from functools import partial
import optax
from tqdm import tqdm
import wandb
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)
rng = jax.random.PRNGKey(42)

# Configuration
DATA_DIR = Path("data")
OUTPUT_DIR = Path("output/block2vec_")
NPY_FILE = DATA_DIR / "block_ids_32_32.npy"
EMBEDDING_DIM = 8  # Size of block embeddings
MAX_WINDOW_SIZE = 17  # Maximum context window size
NEGATIVE_SAMPLES = 5  # Number of negative samples per positive pair
SUBSAMPLE_THRESHOLD = 1e-5  # Subsampling threshold for frequent blocks
LEARNING_RATE = 0.025  # Initial learning rate
BATCH_SIZE = 1024  # Batch size for training
MAX_STEPS = 5000  # Maximum training steps
WANDB_PROJECT = "projectdl"  # Wandb project name
WANDB_GROUP = "block2vec"  # Wandb group name
WANDB_TAG = ["baseline"]  # Wandb tags
CHECKPOINT_INTERVAL = 10  # Save embeddings every N steps
TSNE_SUBSAMPLE = 30  # Number of blocks to visualize (set to None for all)

if not OUTPUT_DIR.exists():
    OUTPUT_DIR.mkdir(parents=True)

def load_data():
    """Load block IDs, ID-to-SNBT mapping, and SNBT counts."""
    block_ids = np.load(NPY_FILE)  # Shape: (n_samples, x_len, y_len, z_len)
    with open(DATA_DIR / "id_to_snbt.json", "r") as f:
        id_to_snbt = json.load(f)
        id_to_snbt = {int(k): v for k, v in id_to_snbt.items()}  # Ensure keys are integers
    with open(DATA_DIR / "snbt_counts.json", "r") as f:
        snbt_counts = json.load(f)
    
    # Create block ID to count mapping
    block_counts = {bid: snbt_counts[snbt] for bid, snbt in id_to_snbt.items()}
    total_count = sum(block_counts.values())
    
    # Compute subsampling probabilities
    vocab_size = len(id_to_snbt)
    block_probs = {bid: count / total_count for bid, count in block_counts.items()}
    subsample_probs = {
        bid: 1 - np.sqrt(SUBSAMPLE_THRESHOLD / (prob + 1e-10))
        for bid, prob in block_probs.items()
    }
    
    # Compute log-uniform sampling probabilities (Zipfian)
    sorted_blocks = sorted(block_counts.items(), key=lambda x: x[1], reverse=True)
    block_ranks = {bid: i + 1 for i, (bid, _) in enumerate(sorted_blocks)}
    log_uniform_probs = np.array([np.log(1 + 1 / block_ranks[bid]) for bid in range(vocab_size)])
    log_uniform_probs /= log_uniform_probs.sum()
    
    return block_ids, id_to_snbt, block_counts, subsample_probs, vocab_size, log_uniform_probs

def data_generator(block_ids, subsample_probs, log_uniform_probs, vocab_size, max_window_size):
    """Infinite data generator yielding (target_id, context_id, neg_context_ids)."""
    n_samples, x_len, y_len, z_len = block_ids.shape
    while True:
        sample_idx = random.randint(0, n_samples - 1)
        x = random.randint(0, x_len - 1)
        y = random.randint(0, y_len - 1)
        z = random.randint(0, z_len - 1)
        
        chunk = block_ids[sample_idx]
        target_id = int(chunk[x, y, z])
        
        if random.random() < subsample_probs.get(target_id, 0):
            continue
        
        current_window_size = random.randint(1, max_window_size)
        
        axis_range = np.arange(-current_window_size, current_window_size + 1, dtype=np.int32)
        dx_grid, dy_grid, dz_grid = np.meshgrid(axis_range, axis_range, axis_range, indexing='ij')
        rel_coords = np.stack([dx_grid.ravel(), dy_grid.ravel(), dz_grid.ravel()], axis=-1)
        is_not_zero_offset = np.any(rel_coords != 0, axis=1)
        rel_coords = rel_coords[is_not_zero_offset]

        abs_coords_x = x + rel_coords[:, 0]
        abs_coords_y = y + rel_coords[:, 1]
        abs_coords_z = z + rel_coords[:, 2]

        valid_mask = (
            (abs_coords_x >= 0) & (abs_coords_x < x_len) &
            (abs_coords_y >= 0) & (abs_coords_y < y_len) &
            (abs_coords_z >= 0) & (abs_coords_z < z_len)
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
        context_id = random.choice(context_ids_list)

        neg_context_ids = np.random.choice(
            vocab_size, size=NEGATIVE_SAMPLES, p=log_uniform_probs, replace=True
        )
        
        yield target_id, context_id, neg_context_ids.astype(np.int32)

def init_model(vocab_size, embedding_dim, rng):
    """Initialize embedding matrices."""
    rng, embed_rng = jax.random.split(rng)
    target_embeddings = jax.random.normal(embed_rng, (vocab_size, embedding_dim)) * 0.01
    context_embeddings = jax.random.normal(embed_rng, (vocab_size, embedding_dim)) * 0.01
    return {"target": target_embeddings, "context": context_embeddings}

@partial(jax.jit)
def nce_loss(params, target_ids, context_ids, neg_context_ids):
    """Compute NCE loss for a batch."""
    target_embed = params["target"][target_ids]
    context_embed = params["context"][context_ids]
    neg_context_embed = params["context"][neg_context_ids]
    
    pos_logits = jnp.sum(target_embed * context_embed, axis=1)
    pos_loss = -jnp.log(jax.nn.sigmoid(pos_logits) + 1e-10)  # Changed jnp.sigmoid to jax.nn.sigmoid
    
    neg_logits = jnp.sum(
        target_embed[:, None, :] * neg_context_embed, axis=2
    )
    neg_loss = -jnp.sum(jnp.log(jax.nn.sigmoid(-neg_logits) + 1e-10), axis=1)  # Changed jnp.sigmoid to jax.nn.sigmoid
    
    return jnp.mean(pos_loss + neg_loss)

@partial(jax.jit, static_argnums=(5,))
def update_step(params, opt_state, target_ids, context_ids, neg_context_ids, optimizer):
    """Perform one optimization step."""
    loss, grads = jax.value_and_grad(nce_loss)(params, target_ids, context_ids, neg_context_ids)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

def visualize_embeddings(embeddings, id_to_snbt, block_counts, vocab_size, step=None):
    """Project embeddings to 2D using t-SNE, save as SVG, and log to wandb."""
    
    if TSNE_SUBSAMPLE is not None and vocab_size > TSNE_SUBSAMPLE:
        top_blocks = sorted(block_counts.items(), key=lambda x: x[1], reverse=True)[:TSNE_SUBSAMPLE]
        indices = [bid for bid, _ in top_blocks]
        sub_embeddings = embeddings[np.array(indices)] # Convert list to NumPy array for indexing
        sub_id_to_snbt = {i: id_to_snbt[i] for i in indices}
        sub_vocab_size = len(indices)
    else:
        sub_embeddings = embeddings
        sub_id_to_snbt = id_to_snbt
        sub_vocab_size = vocab_size
        indices = list(range(vocab_size))
    
    # Run t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=10, max_iter=1000)
    embeddings_2d = tsne.fit_transform(sub_embeddings)  # Shape: (sub_vocab_size, 2)
    
    # Create scatter plot with matplotlib
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=embeddings_2d[:, 0], y=embeddings_2d[:, 1], alpha=0.6)
    
    # Add labels for a subset of points to avoid clutter
    for i in range(min(50, sub_vocab_size)):  # Label up to 50 blocks
        block_name = sub_id_to_snbt[indices[i]].split(':')[-1].split('[')[0]
        plt.text(embeddings_2d[i, 0], embeddings_2d[i, 1], block_name, fontsize=6)
    
    title = f"t-SNE Projection of Block Embeddings (Step {step})" if step is not None else "t-SNE Projection of Final Block Embeddings"
    plt.title(title)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    
    svg_path = OUTPUT_DIR / f"tsne_step_{step}.svg" if step is not None else OUTPUT_DIR / "tsne_final.svg"
    plt.savefig(svg_path, format="svg", bbox_inches="tight")
    png_path = OUTPUT_DIR / f"tsne_step_{step}.png" if step is not None else OUTPUT_DIR / "tsne_final.png"
    plt.savefig(png_path, format="png", bbox_inches="tight", dpi=300)
    plt.close()
    
    # Log to wandb
    wandb.log({
        f"tsne_step_{step if step is not None else 'final'}": wandb.Image(str(png_path))
    })

def train_model(data_gen, vocab_size, embedding_dim, max_steps, id_to_snbt, block_counts, rng):
    """Train the block2vec model with infinite data generator."""
    wandb.init(
        project=WANDB_PROJECT,
        group=WANDB_GROUP,
        config={
            "embedding_dim": EMBEDDING_DIM,
            "max_window_size": MAX_WINDOW_SIZE,
            "negative_samples": NEGATIVE_SAMPLES,
            "subsample_threshold": SUBSAMPLE_THRESHOLD,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "max_steps": MAX_STEPS,
            "tsne_subsample": TSNE_SUBSAMPLE,
        },
        tags=WANDB_TAG
    )
    
    params = init_model(vocab_size, embedding_dim, rng)
    optimizer = optax.adam(LEARNING_RATE)
    opt_state = optimizer.init(params)
    
    batch_target_ids = []
    batch_context_ids = []
    batch_neg_context_ids = []
    step = 0
    
    with tqdm(total=max_steps, desc="Training") as pbar:
        for target_id, context_id, neg_context_ids in data_gen:
            batch_target_ids.append(target_id)
            batch_context_ids.append(context_id)
            batch_neg_context_ids.append(neg_context_ids)
            
            if len(batch_target_ids) == BATCH_SIZE:
                target_ids = jnp.array(batch_target_ids, dtype=jnp.int32)
                context_ids = jnp.array(batch_context_ids, dtype=jnp.int32)
                neg_context_ids = jnp.array(batch_neg_context_ids, dtype=jnp.int32)
                
                params, opt_state, loss = update_step(
                    params, opt_state, target_ids, context_ids, neg_context_ids, optimizer
                )
                
                wandb.log({"step": step, "loss": loss.item()})
                
                if step % CHECKPOINT_INTERVAL == 0 and step > 0:
                    np.save(OUTPUT_DIR / f"block_embeddings_step_{step}.npy", params["target"])
                    visualize_embeddings(params["target"], id_to_snbt, block_counts, vocab_size, step=step)
                
                batch_target_ids = []
                batch_context_ids = []
                batch_neg_context_ids = []
                step += 1
                pbar.update(1)
                
                if step >= max_steps:
                    break
    
    # Save and log final embeddings
    np.save(OUTPUT_DIR / "block_embeddings.npy", params["target"]) 
    visualize_embeddings(params["target"], id_to_snbt, block_counts, vocab_size)
    
    wandb.finish()
    return params["target"]

def main():
    """Main function to run training."""
    print("Loading data...")
    block_ids, id_to_snbt, block_counts, subsample_probs, vocab_size, log_uniform_probs = load_data()
    
    print("Initializing data generator...")
    data_gen = data_generator(
        block_ids, subsample_probs, log_uniform_probs, vocab_size, MAX_WINDOW_SIZE
    )
    
    print(f"Training model for {MAX_STEPS} steps...")
    embeddings = train_model(
        data_gen, vocab_size, EMBEDDING_DIM, MAX_STEPS, id_to_snbt, block_counts, rng
    )
    print("Training complete!")

if __name__ == "__main__":
    main()