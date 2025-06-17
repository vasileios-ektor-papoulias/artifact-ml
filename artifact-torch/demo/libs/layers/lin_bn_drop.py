import torch
import torch.nn as nn


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

    @property
    def in_dim(self) -> int:
        return self._linear.in_features

    @property
    def out_dim(self) -> int:
        return self._linear.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._dropout(self._bn(self._activation(self._linear(x))))
