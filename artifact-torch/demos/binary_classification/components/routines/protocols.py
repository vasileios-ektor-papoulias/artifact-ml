from dataclasses import dataclass
from typing import Optional

import torch
from artifact_torch.base.model.io import ModelInput, ModelOutput
from artifact_torch.core.model.classifier import ClassificationParams


class DemoModelInput(ModelInput):
    pass


class DemoModelOutput(ModelOutput):
    t_loss: Optional[torch.Tensor]


@dataclass
class DemoClassificationParams(ClassificationParams):
    threshold: float
