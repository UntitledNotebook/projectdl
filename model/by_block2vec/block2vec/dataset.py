import numpy as np
import torch
from torch.utils.data import Dataset
from itertools import product

class CustomBlock2VecDataset(Dataset):
    def __init__(self, npy_dataset, neighbor_radius=1):
        
        super().__init__()
        self.dataset = npy_dataset
        self.neighbor_radius = neighbor_radius

        self.block_indices = set()
        for i in range(self.dataset.shape[0]):
            voxel_data = self.dataset[i]
            unique_indices = np.unique(voxel_data)
            self.block_indices.update(unique_indices)

        self.block_frequency = {}
        for i in range(self.dataset.shape[0]):
            voxel_data = self.dataset[i] 
            unique, counts = np.unique(voxel_data, return_counts=True)
            for idx, count in zip(unique, counts):
                self.block_frequency[idx] = self.block_frequency.get(idx, 0) + count

        self._init_discards()

        r = self.neighbor_radius
        self.max_neighbors = (2*r+1)**3 - 1 

    def _init_discards(self):
        t = 0.001
        token_frequencies = list(self.block_frequency.values())
        f = np.array(token_frequencies) / sum(token_frequencies)
        indices = list(self.block_frequency.keys())
        self.idx_to_discard_idx = {idx: i for i, idx in enumerate(indices)}
        self.discards = 1.0 - (np.sqrt(f / t) + 1) * (t / f)

    def __getitem__(self, index):
        sample_idx = index // (16*16*16)
        position_idx = index % (16*16*16)

        z = position_idx % 16
        y = (position_idx // 16) % 16
        x = position_idx // (16*16)

        voxel_data = self.dataset[sample_idx]

        target_block = voxel_data[x, y, z]

        discard_idx = self.idx_to_discard_idx.get(target_block, 0)
        if discard_idx < len(self.discards) and np.random.rand() < self.discards[discard_idx]:
            return self.__getitem__(np.random.randint(self.__len__()))

        context = self._get_neighbors(voxel_data, x, y, z)

        return target_block, context

    def _get_neighbors(self, voxel_data, x, y, z):
        neighbors = []
        r = self.neighbor_radius

        for dx, dy, dz in product(range(-r, r+1), repeat=3):
            if dx == 0 and dy == 0 and dz == 0:
                continue

            nx, ny, nz = x + dx, y + dy, z + dz

            if 0 <= nx < 16 and 0 <= ny < 16 and 0 <= nz < 16:
                neighbors.append(voxel_data[nx, ny, nz])

        if len(neighbors) < self.max_neighbors:
            neighbors.extend([-1] * (self.max_neighbors - len(neighbors)))
        elif len(neighbors) > self.max_neighbors:
            neighbors = neighbors[:self.max_neighbors]

        return np.array(neighbors)

    def __len__(self):
        return self.dataset.shape[0] * 16 * 16 * 16