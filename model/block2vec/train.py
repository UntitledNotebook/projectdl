import os
import torch
import numpy as np
import pytorch_lightning as pl
from block2vec import CustomBlock2Vec
from dataset import CustomBlock2VecDataset

def main():
    # 检查 GPU 状态
    print("CUDA Available:", torch.cuda.is_available())

    # 设置矩阵乘法精度
    torch.set_float32_matmul_precision("medium")

    # 从 npy 文件加载数据集，注意设置 allow_pickle=True
    npy_data = np.load("random_voxel_dataset.npy", allow_pickle=True)
    # 保证数据为列表形式
    dataset_list = list(npy_data)

    # 使用较大的 batch size（2560），并相应线性放缩学习率
    block2vec = CustomBlock2Vec(
        hf_dataset=dataset_list,
        emb_dimension=32,
        batch_size=2560,
        learning_rate=0.01,
        epochs=30,
        output_path="output/block2vec",
        neighbor_radius=2
    )

    # 实例化 Trainer，使用 GPU 加速与混合精度训练
    trainer = pl.Trainer(
        max_epochs=block2vec.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=torch.cuda.device_count() if torch.cuda.is_available() else 1,
        precision="16-mixed" if torch.cuda.is_available() else 32,
    )
    trainer.fit(block2vec)

if __name__ == "__main__":
    main()