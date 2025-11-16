from dataclasses import dataclass
from typing import Optional

import torch
from artifact_torch.nn import ModelInput, ModelOutput
from artifact_torch.table_comparison import GenerationParams


class WorkflowInput(ModelInput):
    pass


class WorkflowOutput(ModelOutput):
    t_loss: Optional[torch.Tensor]


@dataclass
class WorkflowGenerationParams(GenerationParams):
    n_records: int
    temperature: float
