from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict, Generic, List, Optional, Sequence, TypeVar

import torch.nn as nn
from artifact_experiment.base.entities.data_split import DataSplit
from artifact_experiment.base.tracking.client import TrackingClient
from torch.utils.hooks import RemovableHandle

from artifact_torch.base.components.callbacks.periodic import (
    PeriodicCallbackResources,
    PeriodicTrackingCallback,
)
from artifact_torch.base.model.base import Model
from artifact_torch.base.model.io import ModelInput

ModelTCov = TypeVar("ModelTCov", bound=Model, covariant=True)
ModelInputTCov = TypeVar("ModelInputTCov", bound=ModelInput, covariant=True)


@dataclass
class HookCallbackResources(PeriodicCallbackResources, Generic[ModelTCov]):
    model: ModelTCov


ModelTContr = TypeVar("ModelTContr", bound=Model, contravariant=True)
CacheDataT = TypeVar("CacheDataT")
HookResultT = TypeVar("HookResultT")


class HookCallback(
    PeriodicTrackingCallback[HookCallbackResources[ModelTContr], CacheDataT],
    Generic[ModelTContr, CacheDataT, HookResultT],
):
    def __init__(self, period: int, data_split: Optional[DataSplit] = None):
        name = self._get_name()
        super().__init__(name=name, period=period, data_split=data_split)
        self._hook_results: Dict[str, List[HookResultT]] = {}
        self._handles: List[RemovableHandle] = []

    @classmethod
    @abstractmethod
    def _get_name(cls) -> str: ...

    @classmethod
    @abstractmethod
    def _get_layers(cls, model: ModelTContr) -> Sequence[nn.Module]: ...

    @classmethod
    @abstractmethod
    def _attach(
        cls, model: ModelTContr, sink: Dict[str, List[HookResultT]], handles: List[RemovableHandle]
    ): ...

    @classmethod
    @abstractmethod
    def _aggregate(cls, hook_results: Dict[str, List[HookResultT]]) -> CacheDataT: ...

    @staticmethod
    @abstractmethod
    def _export(key: str, value: CacheDataT, tracking_client: TrackingClient): ...

    def attach(self, resources: HookCallbackResources[ModelTContr]) -> bool:
        if self._should_trigger(step=resources.step):
            self._attach(model=resources.model, sink=self._hook_results, handles=self._handles)
            return True
        else:
            return False

    def _compute(self, resources: HookCallbackResources[ModelTContr]) -> CacheDataT:
        _ = resources
        result = self._finalize()
        self._detach()
        return result

    def _finalize(self) -> CacheDataT:
        result = self._aggregate(hook_results=self._hook_results)
        self._hook_results.clear()
        self._cache[self._key] = result
        return result

    def _detach(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()
