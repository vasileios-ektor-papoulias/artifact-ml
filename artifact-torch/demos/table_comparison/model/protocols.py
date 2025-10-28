from dataclasses import dataclass

import torch
from artifact_torch.base.model.io import LossOutput, ModelInput
from artifact_torch.core.model.generative import GenerationParams


class TabularVAEInput(ModelInput):
    t_features: torch.Tensor


class TabularVAEOutput(LossOutput):
    t_reconstructions: torch.Tensor
    z_mean: torch.Tensor
    z_log_var: torch.Tensor


@dataclass
class TabularVAEGenerationParams(GenerationParams):
    n_records: int
    temperature: float
