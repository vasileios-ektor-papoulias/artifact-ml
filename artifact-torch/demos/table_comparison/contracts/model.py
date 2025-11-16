from dataclasses import dataclass
from typing import List, Optional

import torch
from artifact_torch.core import ModelInput, ModelOutput
from artifact_torch.table_comparison import GenerationParams


class TabularVAEInput(ModelInput):
    t_features: torch.Tensor


class TabularVAEOutput(ModelOutput):
    ls_t_logits: List[torch.Tensor]
    t_latent_mean: torch.Tensor
    t_latent_log_var: torch.Tensor
    t_loss: Optional[torch.Tensor]


@dataclass
class TabularVAEGenerationParams(GenerationParams):
    n_records: int
    temperature: float
