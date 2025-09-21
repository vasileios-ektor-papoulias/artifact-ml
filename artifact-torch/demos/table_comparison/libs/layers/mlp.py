from typing import List, Type

import torch
import torch.nn as nn
from demos.table_comparison.libs.layers.lin_bn_drop import LinBnDrop


class MLP(nn.Module):
    def __init__(self, ls_layers: List[LinBnDrop]):
        super().__init__()
        self._ls_layers = ls_layers
        self._mlp = nn.Sequential(*self._ls_layers)

    @classmethod
    def build(
        cls: Type["MLP"],
        ls_layer_sizes: List[int],
        leaky_relu_slope: float,
        bn_momentum: float,
        bn_epsilon: float,
        dropout_rate: float,
    ) -> "MLP":
        ls_layers: List[LinBnDrop] = []
        for i in range(len(ls_layer_sizes) - 1):
            in_dim = ls_layer_sizes[i]
            out_dim = ls_layer_sizes[i + 1]
            ls_layers.append(
                LinBnDrop(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    leaky_relu_slope=leaky_relu_slope,
                    bn_momentum=bn_momentum,
                    bn_epsilon=bn_epsilon,
                    dropout_rate=dropout_rate,
                )
            )
        return cls(ls_layers=ls_layers)

    @property
    def ls_layer_sizes(self) -> List[int]:
        return [layer.out_dim for layer in self._ls_layers]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._mlp(x)
