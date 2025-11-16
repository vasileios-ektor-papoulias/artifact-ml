from typing import List, Tuple

import torch
import torch.nn as nn


class MultiFeaturePredictor(nn.Module):
    def __init__(self, input_dim: int, ls_n_cat_n_embd: List[Tuple[int, int]]):
        super().__init__()
        self._heads = nn.ModuleList([nn.Linear(input_dim, n_cat) for n_cat, _ in ls_n_cat_n_embd])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return [head(x) for head in self._heads]
