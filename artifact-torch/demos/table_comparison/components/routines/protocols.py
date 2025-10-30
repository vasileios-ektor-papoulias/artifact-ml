from dataclasses import dataclass
from typing import Optional

import torch
from artifact_torch.base.model.io import ModelInput, ModelOutput
from artifact_torch.core.model.generative import GenerationParams


class DemoModelInput(ModelInput):
    pass


class DemoModelOutput(ModelOutput):
    t_loss: Optional[torch.Tensor]


@dataclass
class DemoGenerationParams(GenerationParams):
    n_records: int
    temperature: float
