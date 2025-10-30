from dataclasses import dataclass
from typing import Optional

import torch
from artifact_torch.base.model.io import ModelInput, ModelOutput
from artifact_torch.core.model.classifier import ClassificationParams


class MLPClassifierInput(ModelInput):
    t_features: torch.Tensor
    t_targets: torch.Tensor


class MLPClassifierOutput(ModelOutput):
    t_logits: torch.Tensor
    t_loss: Optional[torch.Tensor]


@dataclass
class MLPClassificationParams(ClassificationParams):
    threshold: float
