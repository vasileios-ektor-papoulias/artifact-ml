from typing import Optional

import torch
from artifact_torch.base.model.io import ModelInput, ModelOutput


class TabularVAEInput(ModelInput):
    t_features: torch.Tensor


class TabularVAEOutput(ModelOutput):
    t_reconstructions: torch.Tensor
    z_mean: torch.Tensor
    z_log_var: torch.Tensor
    t_loss: Optional[torch.Tensor]
