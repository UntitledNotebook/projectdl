import os
import math
import torch
import torch.optim as optim
import numpy as np
import pytorch_lightning as pl
import pickle
from skipgram import SkipGramModel
from dataset import CustomBlock2VecDataset
from torch.utils.data import DataLoader


class CustomBlock2Vec(pl.LightningModule):
    def __init__(self, hf_dataset, emb_dimension=32, batch_size=256, 
                 learning_rate=1e-3, epochs=30, output_path="output",
                 neighbor_radius=1):
        super().__init__()
        self.save_hyperparameters(ignore=['hf_dataset'])
        
        # 创建数据集
        self.dataset = CustomBlock2VecDataset(
            hf_dataset, 
            neighbor_radius=neighbor_radius
        )
        
        # 计算唯一方块数量
        self.emb_size = len(self.dataset.block_indices)
        
        # 初始化模型
        self.model = SkipGramModel(self.emb_size, emb_dimension)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.output_path = output_path
        
        # 确保输出目录存在
        os.makedirs(output_path, exist_ok=True)
    
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    def training_step(self, batch, *args, **kwargs):
        loss = self.forward(*batch)
        self.log("loss", loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            math.ceil(len(self.dataset) / self.batch_size) * self.epochs
        )
        return [optimizer], [scheduler]
    
    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=os.cpu_count() or 1
        )
    
    def on_train_epoch_end(self):
        self.save_embedding(self.output_path)
    
    def save_embedding(self, output_path):
        embeddings = self.model.target_embeddings.weight
        embeddings = embeddings.cpu().data.numpy()
        
        # 创建索引到嵌入的映射
        embedding_dict = {}
        for block_idx in self.dataset.block_indices:
            if block_idx < len(embeddings):
                embedding_dict[int(block_idx)] = embeddings[block_idx]
        
        # 保存嵌入表示
        with open(os.path.join(output_path, "index_embeddings.pkl"), "wb") as f:
            pickle.dump(embedding_dict, f)
        
        # 也保存为 NumPy 数组以便灵活使用
        np.save(os.path.join(output_path, "embeddings.npy"), embeddings)
        
        return embedding_dict