from typing import Optional, TypedDict

import torch


class ModelIO(TypedDict):
    pass


class ModelInput(ModelIO):
    pass


class ModelOutput(ModelIO):
    t_loss: Optional[torch.Tensor]
