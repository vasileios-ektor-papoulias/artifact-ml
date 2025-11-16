from dataclasses import dataclass
from typing import Optional

import torch
from artifact_torch.binary_classification import ClassificationParams
from artifact_torch.nn import ModelInput, ModelOutput


class WorkflowInput(ModelInput):
    pass


class WorkflowOutput(ModelOutput):
    t_loss: Optional[torch.Tensor]


@dataclass
class WorkflowClassificationParams(ClassificationParams):
    threshold: float
