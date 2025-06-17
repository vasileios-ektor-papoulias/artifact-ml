from typing import Optional, TypedDict

import torch


class ModelIO(TypedDict):
    pass


class ModelInput(ModelIO):
    pass


class SingleTensorInput(ModelInput):
    t_in: torch.Tensor


class LabeledTensorInput(SingleTensorInput):
    t_labels: Optional[torch.Tensor]


class ModelOutput(ModelIO):
    t_loss: Optional[torch.Tensor]
