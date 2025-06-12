import os
import torch
import numpy as np
import pytorch_lightning as pl
from block2vec import CustomBlock2Vec
from dataset import CustomBlock2VecDataset

def main():
    print("CUDA Available:", torch.cuda.is_available())

    torch.set_float32_matmul_precision("medium")

    dataset_array = np.load("dataset_transformed.npy", allow_pickle=True)

    block2vec = CustomBlock2Vec(
        hf_dataset=dataset_array,
        emb_dimension=6,
        batch_size=32000,
        learning_rate=0.001,
        epochs=30,
        output_path="output/block2vec",
        neighbor_radius=2
    )

    trainer = pl.Trainer(
        max_epochs=block2vec.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=torch.cuda.device_count() if torch.cuda.is_available() else 1,
        precision="16-mixed" if torch.cuda.is_available() else 32,
    )
    trainer.fit(block2vec)

if __name__ == "__main__":
    main()