import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from ddpm import DDPM  # 使用修改后的 DDPM（3D版）
from unet_3D import Unet3D  # 3D Unet，输出通道与输入数据匹配（均为32通道）

# 定义一个随机3D数据集，真实数据每个样本形状为 [32, 16, 16, 16]
class Random3DDataset(Dataset):
    def __init__(self, num_samples=1000):
        super(Random3DDataset, self).__init__()
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        # 生成随机数据作为真实样本，32 表示每个 voxel 的 32 维特征
        sample = torch.randn(32, 16, 16, 16)
        return sample

# 训练函数
def train_ddpm(model, dataloader, optimizer, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, sample in enumerate(dataloader):
            sample = sample.to(device)  # 样本形状: [B, 32, 16, 16, 16]
            optimizer.zero_grad()
            # ddpm_model.forward 内部会:
            # 1. 对输入做前向扩散，生成 x_t；
            # 2. 预测噪声并用 create_mask 限制仅计算内部区域损失；
            loss = model(sample)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/ {epochs}], Step [{i+1}/{len(dataloader)}], Loss: {running_loss/10:.4f}")
                running_loss = 0.0
        torch.save(model.state_dict(), f"ddpm_epoch_{epoch+1}.pth")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 超参数设定
    T = 1000
    betas = (1e-4, 0.02)
    
    # 使用 Unet3D 作为噪声预测网络；这里输入与输出通道数均为 32（与数据一致）
    eps_model = Unet3D(in_channels=32, out_channels=32)
    
    # 使用均方误差作为 loss
    criterion = nn.MSELoss()
    ddpm_model = DDPM(eps_model=eps_model, betas=betas, n_T=T, criterion=criterion)
    ddpm_model.to(device)
    
    # 优化器
    optimizer = optim.Adam(ddpm_model.parameters(), lr=2e-4)
    
    # 构造数据集与 dataloader
    dataset = Random3DDataset(num_samples=1000)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)
    
    epochs = 10
    train_ddpm(ddpm_model, dataloader, optimizer, device, epochs)
    torch.save(ddpm_model.state_dict(), "ddpm_final.pth")

if __name__ == "__main__":
    main()