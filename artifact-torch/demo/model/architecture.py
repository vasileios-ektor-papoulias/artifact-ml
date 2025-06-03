from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import ndarray


class LinBnDrop(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        leaky_relu_slope: float,
        bn_momentum: float,
        bn_epsilon: float,
        dropout_rate: float,
    ):
        super().__init__()
        self._linear = nn.Linear(in_features=in_dim, out_features=out_dim)
        self._activation = nn.LeakyReLU(negative_slope=leaky_relu_slope)
        self._bn = nn.BatchNorm1d(num_features=out_dim, momentum=bn_momentum, eps=bn_epsilon)
        self._dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._dropout(self._bn(self._activation(self._linear(x))))


class VariationalAutoencoder(nn.Module):
    def __init__(
        self,
        ls_encoder_layer_sizes: List[int],
        loss_beta: float,
        leaky_relu_slope: float,
        bn_momentum: float,
        bn_epsilon: float,
        dropout_rate: float,
    ):
        super().__init__()
        if len(ls_encoder_layer_sizes) < 2:
            raise ValueError("ls_encoder_layer_sizes must have at least [input_dim, latent_dim].")
        self._ls_encoder_layer_sizes = ls_encoder_layer_sizes
        self._input_dim = ls_encoder_layer_sizes[0]
        self._latent_dim = ls_encoder_layer_sizes[-1]
        self._loss_beta = loss_beta

        self._encoder = self._build_encoder(
            ls_layer_sizes=self._ls_encoder_layer_sizes,
            leaky_relu_slope=leaky_relu_slope,
            bn_momentum=bn_momentum,
            bn_epsilon=bn_epsilon,
            dropout_rate=dropout_rate,
        )
        self._z_mean, self._z_log_var = self._build_latent_layers(
            last_hidden=self._ls_encoder_layer_sizes[-2], latent_dim=self._latent_dim
        )
        self._decoder = self._build_decoder(
            ls_layer_sizes=list(reversed(self._ls_encoder_layer_sizes)),
            leaky_relu_slope=leaky_relu_slope,
            bn_momentum=bn_momentum,
            bn_epsilon=bn_epsilon,
            dropout_rate=dropout_rate,
        )

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    @property
    def loss_beta(self) -> float:
        return self._loss_beta

    def forward(
        self, t_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        x = t_features
        z_mean, z_log_var = self._encode(x)
        z = self._reparameterize(z_mean=z_mean, z_log_var=z_log_var)
        t_recon = self._decode(z=z)
        t_loss = self._compute_loss(t_recon=t_recon, x=x, z_mean=z_mean, z_log_var=z_log_var)
        return t_recon, z_mean, z_log_var, t_loss

    def generate(self, num_samples: int, use_mean: bool, device: torch.device) -> ndarray:
        self.eval()
        with torch.no_grad():
            z = self._sample_latent(
                num_samples=num_samples,
                latent_dim=self._latent_dim,
                use_mean=use_mean,
                device=device,
            )
            arr_synthetic_flat = self._decode(z).cpu().numpy()
            return arr_synthetic_flat

    def _encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self._encoder(x)
        return self._z_mean(hidden), self._z_log_var(hidden)

    def _reparameterize(self, z_mean: torch.Tensor, z_log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * z_log_var)
        eps = torch.randn_like(std)
        return z_mean + eps * std

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        return self._decoder(z)

    def _compute_loss(
        self,
        t_recon: torch.Tensor,
        x: torch.Tensor,
        z_mean: torch.Tensor,
        z_log_var: torch.Tensor,
    ) -> torch.Tensor:
        recon_loss = F.mse_loss(t_recon, x, reduction="sum")
        kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
        return recon_loss + self._loss_beta * kl_loss

    @staticmethod
    def _sample_latent(
        num_samples: int, latent_dim: int, use_mean: bool, device: torch.device
    ) -> torch.Tensor:
        if use_mean:
            return torch.zeros(num_samples, latent_dim, device=device)
        return torch.randn(num_samples, latent_dim, device=device)

    @staticmethod
    def _build_encoder(
        ls_layer_sizes: List[int],
        leaky_relu_slope: float,
        bn_momentum: float,
        bn_epsilon: float,
        dropout_rate: float,
    ) -> nn.Module:
        layers: List[nn.Module] = []
        ls_sizes = ls_layer_sizes
        for i in range(len(ls_sizes) - 2):
            in_dim = ls_sizes[i]
            out_dim = ls_sizes[i + 1]
            layers.append(
                LinBnDrop(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    leaky_relu_slope=leaky_relu_slope,
                    bn_momentum=bn_momentum,
                    bn_epsilon=bn_epsilon,
                    dropout_rate=dropout_rate,
                )
            )
        encoder = nn.Sequential(*layers)
        return encoder

    @staticmethod
    def _build_latent_layers(last_hidden: int, latent_dim: int) -> Tuple[nn.Module, nn.Module]:
        z_mean = nn.Linear(last_hidden, latent_dim)
        z_log_var = nn.Linear(last_hidden, latent_dim)
        return z_mean, z_log_var

    @staticmethod
    def _build_decoder(
        ls_layer_sizes: List[int],
        leaky_relu_slope: float,
        bn_momentum: float,
        bn_epsilon: float,
        dropout_rate: float,
    ) -> nn.Module:
        layers: List[nn.Module] = []

        for i in range(len(ls_layer_sizes) - 1):
            in_dim = ls_layer_sizes[i]
            out_dim = ls_layer_sizes[i + 1]
            if i == len(ls_layer_sizes) - 2:
                layers.append(nn.Linear(in_dim, out_dim))
            else:
                layers.append(
                    LinBnDrop(
                        in_dim=in_dim,
                        out_dim=out_dim,
                        leaky_relu_slope=leaky_relu_slope,
                        bn_momentum=bn_momentum,
                        bn_epsilon=bn_epsilon,
                        dropout_rate=dropout_rate,
                    )
                )
        decoder = nn.Sequential(*layers)
        return decoder
