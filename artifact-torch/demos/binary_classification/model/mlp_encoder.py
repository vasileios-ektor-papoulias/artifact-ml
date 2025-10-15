from dataclasses import dataclass, field
from typing import List, Type, TypeVar

import torch
import torch.nn as nn

from demos.binary_classification.config.constants import (
    BN_EPSILON,
    BN_MOMENTUM,
    DROPOUT_RATE,
    LEAKY_RELU_SLOPE,
    LS_HIDDEN_SIZES,
)


@dataclass(frozen=True)
class MLPEncoderConfig:
    ls_hidden_sizes: List[int] = field(default_factory=lambda: LS_HIDDEN_SIZES)
    leaky_relu_slope: float = LEAKY_RELU_SLOPE
    bn_momentum: float = BN_MOMENTUM
    bn_epsilon: float = BN_EPSILON
    dropout_rate: float = DROPOUT_RATE


MLPEncoderT = TypeVar("MLPEncoderT", bound="MLPEncoder")


class MLPEncoder(nn.Module):
    def __init__(self, seq: nn.Sequential):
        super().__init__()
        self._seq = seq

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._seq(x)

    @property
    def output_dim(self) -> int:
        for layer in reversed(self._seq):
            if isinstance(layer, nn.Linear):
                return layer.out_features
        raise RuntimeError("MLPEncoder has no Linear layers.")

    @property
    def input_dim(self) -> int:
        for layer in self._seq:
            if isinstance(layer, nn.Linear):
                return layer.in_features
        raise RuntimeError("MLPEncoder has no Linear layers.")

    @classmethod
    def build(
        cls: Type[MLPEncoderT],
        in_dim: int,
        config: MLPEncoderConfig,
    ) -> MLPEncoderT:
        if len(config.ls_hidden_sizes) == 0:
            raise ValueError("ls_hidden_sizes must contain at least one hidden layer.")
        layers: List[nn.Module] = []
        sz_previous = in_dim
        for sz_hidden in config.ls_hidden_sizes:
            layers.append(nn.Linear(sz_previous, sz_hidden))
            layers.append(
                nn.BatchNorm1d(
                    num_features=sz_hidden, eps=config.bn_epsilon, momentum=config.bn_momentum
                )
            )
            layers.append(nn.LeakyReLU(negative_slope=config.leaky_relu_slope, inplace=True))
            if config.dropout_rate > 0:
                layers.append(nn.Dropout(p=config.dropout_rate))
            sz_previous = sz_hidden
        mlp = cls(nn.Sequential(*layers))
        return mlp
