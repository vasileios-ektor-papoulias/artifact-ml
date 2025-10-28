from typing import TypedDict

import torch


class ModelIO(TypedDict):
    pass


class ModelInput(ModelIO):
    pass


class ModelOutput(ModelIO):
    pass


class LossOutput(ModelOutput):
    t_loss: torch.Tensor
