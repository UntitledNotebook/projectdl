import numpy as np
import torch
from torch.utils.data import Dataset
from itertools import product

class CustomBlock2VecDataset(Dataset):
    def __init__(self, npy_dataset, neighbor_radius=1):
        """
        为预处理的体素数据创建 Block2Vec 数据集

        参数:
            npy_dataset: 一个列表，每个元素为包含 'voxels' 键的字典，值为形状 (16,16,16) 的 NumPy 数组
            neighbor_radius: 上下文邻居半径
        """
        super().__init__()
        self.dataset = npy_dataset
        self.neighbor_radius = neighbor_radius

        # 计算所有存在的方块索引
        self.block_indices = set()
        for sample in self.dataset:
            voxel_data = sample['voxels']
            unique_indices = np.unique(voxel_data)
            self.block_indices.update(unique_indices)

        # 计算方块频率（用于负采样）
        self.block_frequency = {}
        for sample in self.dataset:
            voxel_data = sample['voxels']
            unique, counts = np.unique(voxel_data, return_counts=True)
            for idx, count in zip(unique, counts):
                if idx in self.block_frequency:
                    self.block_frequency[idx] += count
                else:
                    self.block_frequency[idx] = count

        # 计算丢弃概率
        self._init_discards()

        # 预计算邻居最大数量
        r = self.neighbor_radius
        self.max_neighbors = (2*r+1)**3 - 1  # 减1是因为排除了中心点

    def _init_discards(self):
        t = 0.001
        token_frequencies = list(self.block_frequency.values())
        f = np.array(token_frequencies) / sum(token_frequencies)
        indices = list(self.block_frequency.keys())
        self.idx_to_discard_idx = {idx: i for i, idx in enumerate(indices)}
        self.discards = 1.0 - (np.sqrt(f / t) + 1) * (t / f)

    def __getitem__(self, index):
        # 确定样本及样本内位置
        sample_idx = index // (16*16*16)
        position_idx = index % (16*16*16)

        # 转换为 3D 坐标
        z = position_idx % 16
        y = (position_idx // 16) % 16
        x = position_idx // (16*16)

        # 获取对应样本数据，并确保 'voxels' 为 NumPy 数组
        sample = self.dataset[sample_idx]
        voxel_data = sample['voxels']

        # 获取目标方块索引
        target_block = voxel_data[x, y, z]

        # 丢弃常见方块（负采样）
        discard_idx = self.idx_to_discard_idx.get(target_block, 0)
        if discard_idx < len(self.discards) and np.random.rand() < self.discards[discard_idx]:
            return self.__getitem__(np.random.randint(self.__len__()))

        # 获取邻居方块
        context = self._get_neighbors(voxel_data, x, y, z)

        return target_block, context

    def _get_neighbors(self, voxel_data, x, y, z):
        neighbors = []
        r = self.neighbor_radius

        for dx, dy, dz in product(range(-r, r+1), repeat=3):
            if dx == 0 and dy == 0 and dz == 0:
                continue  # 排除中心点

            nx, ny, nz = x + dx, y + dy, z + dz

            # 检查边界
            if 0 <= nx < 16 and 0 <= ny < 16 and 0 <= nz < 16:
                neighbors.append(voxel_data[nx, ny, nz])

        # 长度不足时，用 -1 填充
        if len(neighbors) < self.max_neighbors:
            neighbors.extend([-1] * (self.max_neighbors - len(neighbors)))
        elif len(neighbors) > self.max_neighbors:
            neighbors = neighbors[:self.max_neighbors]

        return np.array(neighbors)

    def __len__(self):
        return len(self.dataset) * 16 * 16 * 16