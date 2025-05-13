import torch
import torch.nn as nn

class DoubleConv3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(DoubleConv3d, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)

class UNet3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, base_filters: int = 32):
        """
        in_channels: 输入通道数（例如：原始数据与条件拼接后的总通道数）
        out_channels: 输出通道数（例如：32，表示每个 voxel 的 32 维特征）
        """
        super(UNet3D, self).__init__()
        # 编码部分
        self.inc = DoubleConv3d(in_channels, base_filters)
        self.down1 = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv3d(base_filters, base_filters * 2)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv3d(base_filters * 2, base_filters * 4)
        )
        # 解码部分
        self.up1_upconv = nn.ConvTranspose3d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2)
        self.up1_conv = DoubleConv3d(base_filters * 4, base_filters * 2)
        self.up0_upconv = nn.ConvTranspose3d(base_filters * 2, base_filters, kernel_size=2, stride=2)
        self.up0_conv = DoubleConv3d(base_filters * 2, base_filters)
        # 输出层
        self.outc = nn.Conv3d(base_filters, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, in_channels, 16, 16, 16] 的三维体素数据,
        其中 in_channels 为原始数据与条件拼接后的通道数。
        """
        # 编码器部分
        x1 = self.inc(x)       # [B, base_filters, 16,16,16]
        x2 = self.down1(x1)    # [B, base_filters*2, 8,8,8]
        x3 = self.down2(x2)    # [B, base_filters*4, 4,4,4]

        # 解码器部分
        x_up1 = self.up1_upconv(x3)   # 上采样到 [B, base_filters*2, 8,8,8]
        x_up1 = torch.cat([x_up1, x2], dim=1)  
        x_up1 = self.up1_conv(x_up1)    # [B, base_filters*2, 8,8,8]

        x_up0 = self.up0_upconv(x_up1)  # 上采样到 [B, base_filters, 16,16,16]
        x_up0 = torch.cat([x_up0, x1], dim=1)  
        x_up0 = self.up0_conv(x_up0)    # [B, base_filters, 16,16,16]

        output = self.outc(x_up0)       # 最终输出 [B, out_channels, 16,16,16]
        return output

def create_mask(shape=(1, 1, 16, 16, 16)):
    """
    创建一个 mask，内部 (从第2到第13索引位置) 为 1，边界为 0
    shape: (B, C, D, H, W)
    """
    mask = torch.zeros(shape[2], shape[3], shape[4])
    # 此处内部区域为 12×12×12，边界2个 voxel 保持条件
    mask[2:14, 2:14, 2:14] = 1
    # 扩展到 B, C 维度后返回
    mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, 16,16,16]
    return mask

def apply_mask(generated: torch.Tensor, condition: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    将生成结果的内部区域保留模型预测值，边界区域用 condition 覆盖
    generated, condition: [B, C, 16, 16, 16]
    mask: [1, 1, 16, 16, 16]，其内部 = 1，边界 = 0（会广播到 B, C）
    """
    mask_expanded = mask.expand(generated.size(0), generated.size(1), -1, -1, -1)
    return generated * mask_expanded + condition * (1 - mask_expanded)

if __name__ == '__main__':
    # 假设我们的任务为对一个 16*16*16 的体素块进行生成，每个 voxel 有 32 维（输出通道数为 32）
    # 条件信息已经内嵌在输入数据中（例如 64 通道，其中前 32 通道为待生成区域数据，后 32 通道为条件数据）
    model = UNet3D(in_channels=64, out_channels=32)
    print(model)
    
    # 生成示例输入数据，batch_size=1, 64通道, 16x16x16
    x = torch.randn(1, 64, 16, 16, 16)
    # 分离出条件信息，此处假设 condition 部分对应于 x 中的后32个通道，
    # 或者从原始数据中以其他方式获得边界值，这里为示例直接随机生成
    condition = torch.randn(1, 32, 16, 16, 16)
    
    # 网络生成结果
    generated = model(x)  # [1, 32, 16, 16, 16]
    
    # 创建用于保留边界条件的 mask
    mask = create_mask()  # [1, 1, 16, 16, 16]
    
    # 将生成结果与边界条件结合：内部区域取 generated；边界区域取 condition
    final_output = apply_mask(generated, condition, mask)
    
    print("Final output shape:", final_output.shape) 