from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class BetaVAELoss(nn.Module):
    def __init__(self, beta: float):
        super().__init__()
        self._beta = beta

    @property
    def beta(self) -> float:
        return self._beta

    def forward(
        self,
        ls_t_logits: List[torch.Tensor],
        t_targets: torch.Tensor,
        z_mean: torch.Tensor,
        z_log_var: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = t_targets.size(0)
        recon_loss = sum(
            F.cross_entropy(input=logits, target=t_targets[:, i].long(), reduction="sum")
            for i, logits in enumerate(ls_t_logits)
        )
        kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
        total_loss = recon_loss + self._beta * kl_loss
        return total_loss / batch_size
