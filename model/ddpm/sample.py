import torch
import torch.nn as nn
from ddpm import DDPM  # 使用修改后的 DDPM 模型（3D版）
from unet_3D import Unet3D  # 3D Unet，用于噪声预测

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T = 1000
    betas = (1e-4, 0.02)
    
    # 构造噪声预测网络，与训练时一致
    eps_model = Unet3D(in_channels=32, out_channels=32)
    
    # 构造 DDPM 模型
    ddpm_model = DDPM(eps_model=eps_model, betas=betas, n_T=T, criterion=nn.MSELoss())
    
    # 加载已经保存的模型参数
    ddpm_model.load_state_dict(torch.load("ddpm_final.pth", map_location=device))
    ddpm_model.to(device)
    ddpm_model.eval()
    
    # 创建条件数据，形状为 [B, 32, 16, 16, 16]
    # 注意：条件数据应包含已知边界（例如从真实数据中提取），这里仅示例随机生成
    condition = torch.randn(1, 32, 16, 16, 16).to(device)
    
    # 调用采样函数。size 参数为 (channels, 16, 16, 16)
    sampled = ddpm_model.sample(n_sample=1, size=(32, 16, 16, 16), device=device, condition=condition)
    
    # 保存采样结果（可以进一步处理或可视化）
    torch.save(sampled, "sampled_output.pth")
    print("Sample saved to sampled_output.pth")
    
if __name__ == "__main__":
    main()