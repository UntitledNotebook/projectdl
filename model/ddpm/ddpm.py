"""
Extremely Minimalistic Implementation of DDPM

https://arxiv.org/abs/2006.11239

Everything is self contained. (Except for pytorch and torchvision... of course)

run it with `python superminddpm.py`
"""

from typing import Dict, Tuple
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from unet_3D import create_mask, apply_mask


def ddpm_schedules(beta1: float, beta2: float, T: int) -> Dict[str, torch.Tensor]:
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }


# blk = lambda ic, oc: nn.Sequential(
#     nn.Conv2d(ic, oc, 7, padding=3),
#     nn.BatchNorm2d(oc),
#     nn.LeakyReLU(),
# )


# class DummyEpsModel(nn.Module):
#     """
#     This should be unet-like, but let's don't think about the model too much :P
#     Basically, any universal R^n -> R^n model should work.
#     """

#     def __init__(self, n_channel: int) -> None:
#         super(DummyEpsModel, self).__init__()
#         self.conv = nn.Sequential(  # with batchnorm
#             blk(n_channel, 64),
#             blk(64, 128),
#             blk(128, 256),
#             blk(256, 512),
#             blk(512, 256),
#             blk(256, 128),
#             blk(128, 64),
#             nn.Conv2d(64, n_channel, 3, padding=1),
#         )

#     def forward(self, x, t) -> torch.Tensor:
#         # Lets think about using t later. In the paper, they used Tr-like positional embeddings.
#         return self.conv(x)


class DDPM(nn.Module):
    def __init__(
        self,
        eps_model: nn.Module,
        betas: Tuple[float, float],
        n_T: int,
        criterion: nn.Module = nn.MSELoss(),
    ) -> None:
        super(DDPM, self).__init__()
        self.eps_model = eps_model

        # register_buffer allows us to freely access these tensors by name. It helps device placement.
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.criterion = criterion

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        对输入 x 进行前向扩散，再利用 eps_model 预测噪声，
        仅对内部区域计算损失
        """
        _ts = torch.randint(1, self.n_T, (x.shape[0],)).to(x.device)
        eps = torch.randn_like(x)
        
        # x_t = sqrt(ab) * x + sqrt(1-ab) * eps （公式保持不变）
        x_t = (
            self.sqrtab[_ts, None, None, None, None] * x +
            self.sqrtmab[_ts, None, None, None, None] * eps
        )
        # 模型预测噪声
        eps_pred = self.eps_model(x_t, _ts / self.n_T)
        
        # 创建 mask（假设你采用全局的 create_mask 函数）
        mask = create_mask(x.shape)  # mask shape: [1, 1, 16,16,16]
        mask = mask.to(x.device)
        # 将 mask 扩展到与 x 相同的通道数（例如 x 有 32 个通道）
        mask = mask.expand(x.shape[0], x.shape[1], -1, -1, -1)
        
        # 仅计算内部区域的 MSE 损失
        loss = self.criterion(eps * mask, eps_pred * mask)
        return loss

    def sample(self, n_sample: int, size, device, condition: torch.Tensor) -> torch.Tensor:
        """
        增加参数 condition：条件数据，形状为 [B, C, 16,16,16]（已知边界部分）
        size: (channels, 16,16,16)
        """
        x_i = torch.randn(n_sample, *size).to(device)  # 起始噪声 x_T ~ N(0,1)
        mask = create_mask((1, 1, 16, 16, 16)).to(device)
        
        # 采样：逆扩散过程
        for i in range(self.n_T, 0, -1):
            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0
            eps = self.eps_model(x_i, i / self.n_T)
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )
            # 每一步采样后，确保边界条件不变
            x_i = apply_mask(x_i, condition, mask)
        
        return x_i


# def train_mnist(n_epoch: int = 100, device="cuda:0") -> None:

#     ddpm = DDPM(eps_model=DummyEpsModel(1), betas=(1e-4, 0.02), n_T=1000)
#     ddpm.to(device)

#     tf = transforms.Compose(
#         [transforms.ToTensor(), transforms.Normalize((0.5,), (1.0))]
#     )

#     dataset = MNIST(
#         "./data",
#         train=True,
#         download=True,
#         transform=tf,
#     )
#     dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=20)
#     optim = torch.optim.Adam(ddpm.parameters(), lr=2e-4)

#     for i in range(n_epoch):
#         ddpm.train()

#         pbar = tqdm(dataloader)
#         loss_ema = None
#         for x, _ in pbar:
#             optim.zero_grad()
#             x = x.to(device)
#             loss = ddpm(x)
#             loss.backward()
#             if loss_ema is None:
#                 loss_ema = loss.item()
#             else:
#                 loss_ema = 0.9 * loss_ema + 0.1 * loss.item()
#             pbar.set_description(f"loss: {loss_ema:.4f}")
#             optim.step()

#         ddpm.eval()
#         with torch.no_grad():
#             xh = ddpm.sample(16, (1, 28, 28), device)
#             grid = make_grid(xh, nrow=4)
#             save_image(grid, f"./contents/ddpm_sample_{i}.png")

#             # save model
#             torch.save(ddpm.state_dict(), f"./ddpm_mnist.pth")


# if __name__ == "__main__":
#     train_mnist()
 