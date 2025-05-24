from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, TypeVar

import torch
import torch.nn as nn

from artifact_torch.base.model.io import (
    ModelInput,
    ModelOutput,
)

ModelInputT = TypeVar("ModelInputT", bound="ModelInput")
ModelOutputT = TypeVar("ModelOutputT", bound="ModelOutput")


class Model(ABC, nn.Module, Generic[ModelInputT, ModelOutputT]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._device = torch.device("cpu")

    def __call__(self, model_input: ModelInputT, *args, **kwargs) -> ModelOutputT:
        return super().__call__(model_input, *args, **kwargs)

    @property
    def params(self) -> Dict[str, Any]:
        return self.state_dict().copy()

    @property
    def param_count(self) -> Dict[str, int]:
        return self._get_param_count(model=self)

    @property
    def device(self) -> torch.device:
        return self._device

    @device.setter
    def device(self, device: torch.device):
        self._move_to_device(device=device)

    @abstractmethod
    def forward(self, model_input: ModelInputT, *args, **kwargs) -> ModelOutputT: ...

    def load_params(self, params: Dict[str, Any]):
        self.load_state_dict(state_dict=params)
        self._move_to_device(self._device)

    def _move_to_device(self, device: torch.device):
        self._device = device
        self.to(device)

    @staticmethod
    def _get_param_count(
        model: nn.Module,
    ) -> Dict[str, int]:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params
        dict_param_count = {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "non_trainable_parameters": non_trainable_params,
        }
        return dict_param_count
