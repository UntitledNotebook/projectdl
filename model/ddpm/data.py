import torch
import torch.utils.data
import numpy as np
from scipy.spatial.distance import cdist
import os
import logging
from typing import Tuple, Optional 

class BlockDataset(torch.utils.data.Dataset):
    """
    Dataset for Minecraft voxel data.
    Loads the entire block ID dataset into RAM and their embeddings.
    Embedding lookup and standardization are done on-the-fly in __getitem__.
    Returns the voxel embedding tensor directly.
    """
    def __init__(self, data_path: str, embedding_path: str, voxel_shape: Tuple[int, int, int],
                 embedding_dim: int, standardize_embeddings: bool = True):
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        if not os.path.exists(embedding_path):
            raise FileNotFoundError(f"Embedding file not found: {embedding_path}")

        logging.info(f"Loading entire block ID data from {data_path} into RAM...")
        self.raw_block_ids = np.load(data_path).astype(np.int64)
        
        if self.raw_block_ids.ndim != 4 or self.raw_block_ids.shape[1:] != voxel_shape:
            raise ValueError(
                f"Block ID data shape mismatch. Expected (N, {voxel_shape[0]}, {voxel_shape[1]}, {voxel_shape[2]}), "
                f"got {self.raw_block_ids.shape}"
            )
        self.num_samples = self.raw_block_ids.shape[0]
        logging.info(f"Successfully loaded {self.num_samples} block ID samples into RAM. Shape: {self.raw_block_ids.shape}")

        block_embeddings_np = np.load(embedding_path).astype(np.float32)
        if block_embeddings_np.ndim != 2 or block_embeddings_np.shape[1] != embedding_dim:
            raise ValueError(
                f"Embedding shape mismatch. Expected (num_ids, {embedding_dim}), "
                f"got {block_embeddings_np.shape}"
            )
        self.block_embeddings_torch = torch.from_numpy(block_embeddings_np)
        
        self.standardize_embeddings = standardize_embeddings
        if self.standardize_embeddings:
            self.embedding_mean = self.block_embeddings_torch.mean(dim=0)
            self.embedding_std = self.block_embeddings_torch.std(dim=0) + 1e-8
            self.embedding_mean = self.embedding_mean.view(1, 1, 1, embedding_dim) # For (D,H,W,E)
            self.embedding_std = self.embedding_std.view(1, 1, 1, embedding_dim)   # For (D,H,W,E)
            logging.info("Embeddings will be standardized on-the-fly for training data.")
        
        logging.info(f"Initialized BlockDataset with {self.num_samples} samples.")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        item_block_ids_np = self.raw_block_ids[idx]
        item_block_ids_torch = torch.from_numpy(item_block_ids_np)
        
        try:
            item_voxel_embeddings = self.block_embeddings_torch[item_block_ids_torch] # (D,H,W,E)
        except IndexError as e:
            max_id_in_sample = item_block_ids_torch.max().item()
            logging.error(
                f"IndexError during embedding lookup for sample {idx}. Max ID: {max_id_in_sample}. "
                f"Embeddings table size: {self.block_embeddings_torch.shape[0]}. Error: {e}"
            )
            raise e

        if self.standardize_embeddings:
            # Mean/std are already shaped (1,1,1,E) for broadcasting with (D,H,W,E)
            item_voxel_embeddings = (item_voxel_embeddings - self.embedding_mean.to(item_voxel_embeddings.device)) / \
                                    self.embedding_std.to(item_voxel_embeddings.device)
        
        item_voxel_embeddings = item_voxel_embeddings.permute(3, 0, 1, 2) # (E, D, H, W)
        
        return item_voxel_embeddings

def get_dataset(
    data_path: str, embedding_path: str, batch_size: int, voxel_shape: Tuple[int, int, int],
    embedding_dim: int, standardize_embeddings: bool = True, num_workers: int = 0, shuffle: bool = True
) -> torch.utils.data.DataLoader:
    dataset = BlockDataset(
        data_path, embedding_path, voxel_shape, embedding_dim, standardize_embeddings
    )
    pin_memory_enabled = torch.cuda.is_available()
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        pin_memory=pin_memory_enabled, drop_last=True 
    )
    logging.info(
        f"DataLoader created: batch_size={batch_size}, shuffle={shuffle}, "
        f"num_workers={num_workers}, pin_memory={pin_memory_enabled}"
    )
    return dataloader

def get_processed_block_embedding_table( # Renamed for clarity from get_block_embeddings_and_stats
    embedding_path: str, 
    embedding_dim: int,
    standardize_table: bool = True, # If True, the entire table is standardized
    device: torch.device = torch.device('cpu')
) -> torch.Tensor:
    """
    Loads the full block embedding table and returns it, optionally standardized.
    This is useful for creating a reference table for `map_embeddings_to_ids`
    if the model outputs standardized embeddings.

    Args:
        embedding_path (str): Path to the .npy file with Block2Vec embeddings.
        embedding_dim (int): Expected dimensionality of embeddings.
        standardize_table (bool): If True, the entire returned embedding table
                                  will be standardized using its own mean and std.
        device (torch.device): Device to load tensors onto.

    Returns:
        torch.Tensor: The full block embedding table, processed as specified.
                      Shape: (num_unique_ids, embedding_dim).
    """
    if not os.path.exists(embedding_path):
        raise FileNotFoundError(f"Embedding file not found: {embedding_path}")
    
    original_block_embeddings_np = np.load(embedding_path).astype(np.float32)
    if original_block_embeddings_np.ndim != 2 or original_block_embeddings_np.shape[1] != embedding_dim:
        raise ValueError(
            f"Embedding shape mismatch. Expected (num_ids, {embedding_dim}), "
            f"got {original_block_embeddings_np.shape}"
        )

    embedding_table = torch.from_numpy(original_block_embeddings_np).to(device)
    
    if standardize_table:
        # Standardize the whole table using its own mean and std
        mean = torch.mean(embedding_table, dim=0, keepdim=True)
        std = torch.std(embedding_table, dim=0, keepdim=True) + 1e-8
        processed_embedding_table = (embedding_table - mean) / std
        return processed_embedding_table
    else:
        return embedding_table # Return the original table if no standardization requested for the table

def map_ids_to_embeddings( # This function seems fine as is
    block_ids_batch: np.ndarray, 
    # This `all_block_embeddings` should be the table in the space model expects (e.g. standardized)
    all_block_embeddings_for_input: torch.Tensor, 
    target_device: torch.device
) -> torch.Tensor:
    """
    Converts a batch of block IDs to their corresponding embeddings for model input.
    The `all_block_embeddings_for_input` tensor should be the embedding table
    processed in the same way as the training data (e.g., standardized if needed).
    """
    original_shape = block_ids_batch.shape
    flat_ids = torch.from_numpy(block_ids_batch.flatten().astype(np.int64)).to(all_block_embeddings_for_input.device)
    
    try:
        embeddings_flat = all_block_embeddings_for_input[flat_ids]
    except IndexError as e:
        max_id = flat_ids.max().item()
        logging.error(
            f"IndexError: map_ids_to_embeddings. Max ID: {max_id}. "
            f"Ref table size: {all_block_embeddings_for_input.shape[0]}. Error: {e}"
        )
        raise e
        
    reshaped_embeddings = embeddings_flat.reshape(*original_shape, -1) # B,D,H,W,E
    return reshaped_embeddings.permute(0, 4, 1, 2, 3).to(target_device) # B,E,D,H,W

def map_embeddings_to_ids(
    predicted_embeddings_batch: torch.Tensor, 
    # This reference table should be in the SAME space as predicted_embeddings_batch
    # If model outputs standardized embeddings, this should be the standardized full embedding table.
    reference_embedding_table: torch.Tensor, 
    metric: str = 'cosine'
) -> np.ndarray:
    """
    Converts a batch of predicted embeddings back to the closest block IDs.
    The comparison is done between `predicted_embeddings_batch` and `reference_embedding_table`.
    Both are expected to be in the same space (e.g., both standardized if applicable).

    Args:
        predicted_embeddings_batch (torch.Tensor): Batch of predicted embeddings from model. (B, E, D, H, W).
        reference_embedding_table (torch.Tensor): The full table of embeddings to compare against.
                                                  Should be in the same space as predictions.
                                                  Shape: (num_unique_ids, E). Should be on CPU for cdist.
        metric (str): Distance metric ('cosine' or 'euclidean').

    Returns:
        np.ndarray: Batch of predicted block IDs. Shape: (B, D, H, W).
    """
    predicted_embeddings_batch_cpu = predicted_embeddings_batch.cpu()
    reference_embedding_table_cpu = reference_embedding_table.cpu() # Ensure on CPU

    predicted_embeddings_permuted = predicted_embeddings_batch_cpu.permute(0, 2, 3, 4, 1) # B,D,H,W,E
    batch_size, D, H, W, E = predicted_embeddings_permuted.shape
    
    predicted_embeddings_flat_np = predicted_embeddings_permuted.reshape(-1, E).numpy() # (B*D*H*W, E)
    reference_table_np = reference_embedding_table_cpu.numpy() # (num_ids, E)

    if metric not in ['cosine', 'euclidean']:
        raise ValueError(f"Unknown distance metric: {metric}. Choose 'cosine' or 'euclidean'.")
        
    distances = cdist(predicted_embeddings_flat_np, reference_table_np, metric=metric) # (B*D*H*W, num_ids)
    closest_ids_flat = np.argmin(distances, axis=1) # (B*D*H*W,)
    
    return closest_ids_flat.reshape(batch_size, D, H, W).astype(np.int32)
