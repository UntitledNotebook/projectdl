from typing import Dict, Tuple
from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.utils import save_image, make_grid

def apply_mask(generated: torch.Tensor, condition: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    
    mask_expanded = mask.expand(generated.size(0), generated.size(1), -1, -1, -1)
    return generated * mask_expanded + condition * (1 - mask_expanded)

def create_mask(shape=(1, 1, 16, 16, 16)):
    
    mask = torch.zeros(shape[2], shape[3], shape[4])
    mask[2:30, 2:30, 2:30] = 1
    mask = mask.unsqueeze(0).unsqueeze(0)
    return mask

def ddpm_schedules( T: int) -> Dict[str, torch.Tensor]:
    

    def f(t):
        s = 0.008
        return torch.cos((t + s) / (1 + s) * np.pi / 2) ** 2

    timesteps = torch.arange(0, T + 1, dtype=torch.float32)
    T_tensor = torch.tensor(T, dtype=torch.float32)

    alpha_bar_t = f(timesteps / T_tensor) / f(torch.tensor(0.0))


    beta_t = torch.clip(1 - alpha_bar_t[1:] / alpha_bar_t[:-1], 0, 0.999)
    
    
    
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,
        "oneover_sqrta": oneover_sqrta,
        "sqrt_beta_t": sqrt_beta_t,
        "alphabar_t": alphabar_t,
        "sqrtab": sqrtab,
        "sqrtmab": sqrtmab,
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,
    }

class DDPM(nn.Module):
    def __init__(
        self,
        eps_model: nn.Module,
        n_T: int,
        criterion: nn.Module = nn.MSELoss(),
    ) -> None:
        super(DDPM, self).__init__()
        self.eps_model = eps_model

        for k, v in ddpm_schedules( n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.criterion = criterion

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        _ts = torch.randint(1, self.n_T, (x.shape[0],)).to(x.device)
        eps = torch.randn_like(x)
        
        x_t = (
            self.sqrtab[_ts, None, None, None, None] * x +
            self.sqrtmab[_ts, None, None, None, None] * eps
        )
        eps_pred = self.eps_model(x_t, _ts / self.n_T)
        
        loss = self.criterion(eps, eps_pred)
        return loss

    def sample(self, n_sample: int, size, device, condition: torch.Tensor) -> torch.Tensor:
        
        with torch.no_grad():
            x_i = torch.randn(n_sample, *size).to(device)
            mask = create_mask((1, 1, 32, 32, 32)).to(device)
            for i in range(self.n_T - 1, 0, -1):
                z = torch.randn(n_sample, *size).to(device) if i > 1 else 0
                t_tensor = torch.full((n_sample,), i / self.n_T, device=device, dtype=torch.float32)
                
                eps = self.eps_model(x_i, t_tensor)
                x_i = (
                    self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                    + self.sqrt_beta_t[i] * z
                )
                x_i = apply_mask(x_i, condition, mask)
        
        return x_i
 