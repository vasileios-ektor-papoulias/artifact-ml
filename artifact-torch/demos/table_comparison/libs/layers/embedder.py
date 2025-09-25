from typing import List, Tuple

import torch
import torch.nn as nn


class MultiFeatureEmbedder(nn.Module):
    def __init__(self, ls_n_cat_n_embd: List[Tuple[int, int]]):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_embeddings=n_cat, embedding_dim=n_embd)
                for n_cat, n_embd in ls_n_cat_n_embd
            ]
        )
        self._total_dim = sum(n_embd for _, n_embd in ls_n_cat_n_embd)

    @property
    def total_dim(self) -> int:
        return self._total_dim

    def forward(self, t_features: torch.Tensor) -> torch.Tensor:
        if len(self.embeddings) == 0:
            return t_features
        embedded = [
            embedding(t_features[:, i].long()) for i, embedding in enumerate(self.embeddings)
        ]
        return torch.cat(embedded, dim=1)
