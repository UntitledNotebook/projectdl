import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class TransformerPositionalEmbedding(nn.Module):
    
    def __init__(self, dimension):
        super(TransformerPositionalEmbedding, self).__init__()
        self.dimension = dimension
        assert dimension % 2 == 0, "嵌入维度必须是偶数"
        self.dense1 = nn.Linear(dimension, dimension * 4)
        self.dense2 = nn.Linear(dimension * 4, dimension)
        self.activation = nn.SiLU()

    def forward(self, time):
        time = time.float()
        half_dim = self.dimension // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(0, half_dim, dtype=torch.float32) / half_dim
        ).to(device=time.device)
        args = time[:, None] * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        embedding = self.activation(self.dense1(embedding))
        embedding = self.dense2(embedding)
        return embedding

class SelfAttention3D(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention3D, self).__init__()
        self.query = nn.Conv3d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv3d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        b, c, d, h, w = x.shape
        q = self.query(x).view(b, -1, d * h * w)
        k = self.key(x).view(b, -1, d * h * w)
        v = self.value(x).view(b, -1, d * h * w)
        attn = torch.softmax(torch.bmm(q.transpose(1, 2), k), dim=-1)
        out = torch.bmm(v, attn.transpose(1, 2)).view(b, c, d, h, w)
        return self.gamma * out + x

class DenseResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels, groups=8):
        super(DenseResnetBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(groups, out_channels)
        self.relu = nn.ReLU(inplace=True)
        if in_channels != out_channels:
            self.residual_1x1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_1x1 = nn.Identity()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_channels, time_channels),
            nn.ReLU(inplace=True),
            nn.Linear(time_channels, out_channels)
        )
        
    def forward(self, x, time_embedding=None):
        residual_x = self.residual_1x1(x)
        x = self.conv1(x)
        x = self.gn1(x)
        if time_embedding is not None:
            time_proj = self.time_mlp(time_embedding)
            x = x + time_proj[:, :, None, None, None].expand(-1, -1, x.shape[2], x.shape[3], x.shape[4])
        x = self.relu(x)
        x = self.conv2(x)
        x = self.gn2(x)
        x = x + residual_x
        x = x * 0.7071
        return self.relu(x)

class UNet3D(nn.Module):
    
    def __init__(self, time_channels=256, block_embedding_dimensions=64):
        super(UNet3D, self).__init__()
        self.time_channels = time_channels
        self.enc1 = DenseResnetBlock(block_embedding_dimensions, 64, time_channels)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.enc2 = DenseResnetBlock(64, 128, time_channels)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.enc3 = DenseResnetBlock(128, 256, time_channels)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.enc4 = DenseResnetBlock(256, 512, time_channels)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.bottleneck = DenseResnetBlock(512, 1024, time_channels)
        self.attn = SelfAttention3D(1024)
        
        self.up1 = nn.ConvTranspose3d(1024, 512, kernel_size=2, stride=2)
        self.dec1 = DenseResnetBlock(512 + 512, 512, time_channels)
        self.up2 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.dec2 = DenseResnetBlock(256 + 256, 256, time_channels)
        self.up3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.dec3 = DenseResnetBlock(128 + 128, 128, time_channels)
        self.up4 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec4 = DenseResnetBlock(64 + 64, 64, time_channels)
        self.final = nn.Conv3d(64, block_embedding_dimensions, kernel_size=1)
        
        self.positional_embed = TransformerPositionalEmbedding(dimension=time_channels)
        
    def forward(self, x, time):
        time_emb = self.positional_embed(time)
        enc1 = self.enc1(x, time_emb)
        enc1_pool = self.pool1(enc1)
        enc2 = self.enc2(enc1_pool, time_emb)
        enc2_pool = self.pool2(enc2)
        enc3 = self.enc3(enc2_pool, time_emb)
        enc3_pool = self.pool3(enc3)
        enc4 = self.enc4(enc3_pool, time_emb)
        enc4_pool = self.pool4(enc4)
        
        bottleneck = self.bottleneck(enc4_pool, time_emb)
        bottleneck = self.attn(bottleneck)
        
        up1 = self.up1(bottleneck)
        up1 = torch.cat((up1, enc4), dim=1)
        dec1 = self.dec1(up1, time_emb)
        up2 = self.up2(dec1)
        up2 = torch.cat((up2, enc3), dim=1)
        dec2 = self.dec2(up2, time_emb)
        up3 = self.up3(dec2)
        up3 = torch.cat((up3, enc2), dim=1)
        dec3 = self.dec3(up3, time_emb)
        up4 = self.up4(dec3)
        up4 = torch.cat((up4, enc1), dim=1)
        dec4 = self.dec4(up4, time_emb)
        final = self.final(dec4)
        return final