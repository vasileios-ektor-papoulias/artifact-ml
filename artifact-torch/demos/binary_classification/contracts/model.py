from dataclasses import dataclass
from typing import Optional

import torch
from artifact_torch.binary_classification import ClassificationParams
from artifact_torch.nn import ModelInput, ModelOutput


class MLPClassifierInput(ModelInput):
    t_features: torch.Tensor
    t_targets: torch.Tensor


class MLPClassifierOutput(ModelOutput):
    t_logits: torch.Tensor
    t_loss: Optional[torch.Tensor]


@dataclass
class MLPClassificationParams(ClassificationParams):
    threshold: float
