from typing import Tuple

import torch
import torch.nn as nn


class DiagonalGaussianLatent(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int):
        super().__init__()
        self._mean = nn.Linear(in_dim, latent_dim)
        self._log_var = nn.Linear(in_dim, latent_dim)

    @property
    def in_dim(self) -> int:
        return self._mean.in_features

    @property
    def latent_dim(self) -> int:
        return self._mean.out_features

    def forward(self, t_hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._mean(t_hidden), self._log_var(t_hidden)

    def reparameterize(self, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def sample(
        self,
        num_samples: int,
        latent_dim: int,
        use_mean: bool,
        device: torch.device,
    ) -> torch.Tensor:
        if use_mean:
            return torch.zeros(num_samples, latent_dim, device=device)
        return torch.randn(num_samples, latent_dim, device=device)
