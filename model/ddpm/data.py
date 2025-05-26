# data.py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import logging, os, time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def floats_to_ids(model_output_tensor: torch.Tensor, bit_length: int) -> np.ndarray:
    """
    Converts model output (float tensor representing bits scaled to ~[-1, 1])
    back to integer block IDs.

    Args:
        model_output_tensor (torch.Tensor): Tensor from the model.
            Expected shape (N, L, X, Y, Z) or (L, X, Y, Z),
            where N is batch_size, L is bit_length (channels).
            Values are floats, where > 0 is considered bit 1, <= 0 is bit 0.
        bit_length (int): The fixed length of the bit representation.

    Returns:
        np.ndarray: NumPy array of integer block IDs.
            Shape (N, X, Y, Z) or (X, Y, Z).
    """
    if isinstance(model_output_tensor, torch.Tensor):
        model_output_numpy = model_output_tensor.detach().cpu().numpy()
    binary_bits = (model_output_numpy > 0).astype(np.uint8)
    powers = 2 ** np.arange(bit_length, dtype=np.uint32)
    
    if binary_bits.ndim == 5: # Batch processing (N, L, X, Y, Z)
        powers_reshaped = powers.reshape(1, -1, 1, 1, 1)
        ids = np.sum(binary_bits * powers_reshaped, axis=1)
    elif binary_bits.ndim == 4: # Single sample processing (L, X, Y, Z)
        powers_reshaped = powers.reshape(-1, 1, 1, 1)
        ids = np.sum(binary_bits * powers_reshaped, axis=0)
    
    return ids.astype(np.int32)

class VoxelDataset(Dataset):
    """
    PyTorch Dataset for voxel data.
    Loads data from an .npy file, converts block IDs to bit representations (optimized),
    scales them to [-1, 1], and provides samples in (Channels, Depth, Height, Width) format.
    """
    def __init__(self, npy_file_path: str, bit_length: int):
        self.npy_file_path = npy_file_path
        self.bit_length = bit_length
        
        raw_data = np.load(npy_file_path)[..., np.newaxis] # (N, X, Y, Z, 1)
        bit_indices = np.arange(self.bit_length, dtype=np.uint8)

        all_bits = (raw_data >> bit_indices) & 1
        self.processed_data = 2.0 * all_bits.astype(np.float32) - 1.0
        self.processed_data = torch.from_numpy(np.transpose(self.processed_data, (0, 4, 1, 2, 3))) # (N, L, X, Y, Z)
        logging.info(f"Loaded {self.processed_data.shape[0]} samples.")


    def __len__(self) -> int:
        return self.processed_data.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.processed_data[idx].detach()

def get_dataloader(
    npy_file_path: str, 
    bit_length: int, 
    batch_size: int, 
    shuffle: bool = True, 
    num_workers: int = 0,
    pin_memory: bool = True
) -> DataLoader:
    dataset = VoxelDataset(npy_file_path=npy_file_path, bit_length=bit_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)