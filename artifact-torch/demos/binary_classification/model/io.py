from typing import Optional

import torch
from artifact_torch.base.model.io import ModelInput, ModelOutput


class MLPClassifierInput(ModelInput):
    t_features: torch.Tensor
    t_targets: torch.Tensor


class MLPClassifierOutput(ModelOutput):
    t_logits: torch.Tensor
    t_loss: Optional[torch.Tensor]
