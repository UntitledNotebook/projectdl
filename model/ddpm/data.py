import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial.distance import cdist
import os

def get_dataset(rng, data_path, embedding_path, batch_size, voxel_shape, embedding_dim, standardize_embeddings=True):
    """Loads and preprocesses Minecraft block data with Block2Vec embeddings.

    Loads block IDs from data_path and maps them to Block2Vec embeddings from
    embedding_path. Optionally standardizes embeddings to zero mean and unit
    variance across the embedding dimension.

    Args:
        rng (jax.random.PRNGKey): Random key for shuffling batches.
        data_path (str): Path to .npy file with block IDs, shape (n_samples, *voxel_shape).
        embedding_path (str): Path to .npy file with Block2Vec embeddings, shape (num_blocks, embedding_dim).
        batch_size (int): Total batch size across all devices.
        voxel_shape (Tuple[int, int, int]): Shape of voxel grid (depth, height, width).
        embedding_dim (int): Dimensionality of Block2Vec embeddings.
        standardize_embeddings (bool): If True, standardize embeddings (default: True).

    Returns:
        callable: Function that returns a batch of preprocessed data with shape
            (num_devices, batch_size_per_device, *voxel_shape, embedding_dim).
    """
    if batch_size % jax.device_count() > 0:
        raise ValueError('Batch size must be divisible by the number of devices')
    batch_size_per_device = batch_size // jax.process_count()
    data = np.load(data_path).astype(np.int32)  # Shape: (n_samples, *voxel_shape)
    embeddings = np.load(embedding_path).astype(np.float32)  # Shape: (num_blocks, embedding_dim)
    data = embeddings[data]  # Shape: (n_samples, *voxel_shape, embedding_dim)
    if standardize_embeddings:
        # Standardize embeddings: zero mean, unit variance across embedding dimension
        mean = embeddings.mean(axis=0, keepdims=True)  # Shape: (1, embedding_dim)
        std = embeddings.std(axis=0, keepdims=True) + 1e-8  # Shape: (1, embedding_dim)
        data = (data - mean) / std
    n_samples = data.shape[0]
    
    def get_batch():
        idx = jax.random.permutation(rng, n_samples)[:batch_size]
        batch = data[idx]
        batch = batch.reshape((jax.local_device_count(), batch_size_per_device, *voxel_shape, embedding_dim))
        return {'voxel': batch}
    
    return get_batch

# Added utility functions for inpainting script
def get_block_embeddings_and_stats(path: str, standardize: bool = True, embedding_dim_val: int = 4):
    """
    Loads block embeddings from a .npy file and optionally computes their mean and std.
    Returns the processed embeddings, original embeddings, mean, and std.
    """
    if not os.path.exists(path): # Ensure os is imported if not already
        raise FileNotFoundError(f"Embedding file not found: {path}")
    all_block_embeddings_original = jnp.array(np.load(path), dtype=jnp.float32)
    
    if standardize:
        embedding_mean = jnp.mean(all_block_embeddings_original, axis=0, keepdims=True)
        embedding_std = jnp.std(all_block_embeddings_original, axis=0, keepdims=True)
        # Add epsilon to std to prevent division by zero
        processed_block_embeddings = (all_block_embeddings_original - embedding_mean) / (embedding_std + 1e-8)
        return processed_block_embeddings, all_block_embeddings_original, embedding_mean, embedding_std
    else:
        # Ensure embedding_dim_val is used if not standardizing to create zero mean and unit std
        embedding_mean = jnp.zeros((1, embedding_dim_val), dtype=jnp.float32)
        embedding_std = jnp.ones((1, embedding_dim_val), dtype=jnp.float32)
        return all_block_embeddings_original, all_block_embeddings_original, embedding_mean, embedding_std

def map_ids_to_embeddings(block_ids_batch: np.ndarray, all_block_embeddings: jnp.ndarray) -> jnp.ndarray:
    """Converts a batch of block IDs to their corresponding embeddings."""
    original_shape = block_ids_batch.shape
    flat_ids = block_ids_batch.flatten()
    
    all_block_embeddings_jax = jnp.asarray(all_block_embeddings)
    embeddings = all_block_embeddings_jax[flat_ids]
    
    return embeddings.reshape(*original_shape, -1)

def map_embeddings_to_ids(predicted_embeddings_batch: jnp.ndarray, all_original_block_embeddings: jnp.ndarray, metric: str = 'euclidean') -> np.ndarray:
    """Converts a batch of predicted embeddings back to block IDs using the specified distance metric."""
    batch_size, D, H, W, E = predicted_embeddings_batch.shape
    predicted_embeddings_flat = predicted_embeddings_batch.reshape(-1, E)
    
    predicted_embeddings_np = np.array(predicted_embeddings_flat)
    all_original_block_embeddings_np = np.array(all_original_block_embeddings)

    if metric == 'cosine':
        distances = cdist(predicted_embeddings_np, all_original_block_embeddings_np, metric='cosine')
    elif metric == 'euclidean':
        distances = cdist(predicted_embeddings_np, all_original_block_embeddings_np, metric='euclidean')
    else:
        raise ValueError(f"Unknown distance metric: {metric}")
        
    closest_ids = np.argmin(distances, axis=1)
    return closest_ids.reshape(batch_size, D, H, W)