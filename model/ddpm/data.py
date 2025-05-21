import jax
import jax.numpy as jnp
import numpy as np

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