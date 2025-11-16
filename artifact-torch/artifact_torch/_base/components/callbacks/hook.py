from abc import abstractmethod
from typing import Dict, Generic, List, Optional, Sequence, TypeVar

import torch
import torch.nn as nn
from artifact_core.typing import (
    Array,
    ArrayCollection,
    ArtifactResult,
    Plot,
    PlotCollection,
    Score,
    ScoreCollection,
)
from artifact_experiment.tracking.spi import TrackingQueueWriter
from torch.utils.hooks import RemovableHandle

from artifact_torch._base.components.callbacks.periodic import PeriodicTrackingCallback
from artifact_torch._base.components.resources.hook import HookCallbackResources
from artifact_torch._base.model.base import Model

ModelTContr = TypeVar("ModelTContr", bound=Model, contravariant=True)
CacheDataTCov = TypeVar("CacheDataTCov", bound=ArtifactResult, covariant=True)
HookResultT = TypeVar("HookResultT")


class HookCallback(
    PeriodicTrackingCallback[HookCallbackResources[ModelTContr], CacheDataTCov],
    Generic[ModelTContr, CacheDataTCov, HookResultT],
):
    def __init__(self, period: int, writer: Optional[TrackingQueueWriter[CacheDataTCov]] = None):
        base_key = self._get_base_key()
        super().__init__(base_key=base_key, period=period, writer=writer)
        self._hook_results: Dict[str, List[HookResultT]] = {}
        self._handles: List[RemovableHandle] = []

    @classmethod
    @abstractmethod
    def _get_base_key(cls) -> str: ...

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
    def _aggregate(cls, hook_results: Dict[str, List[HookResultT]]) -> CacheDataTCov: ...

    def attach(self, resources: HookCallbackResources[ModelTContr]) -> bool:
        if self._should_trigger(step=resources.step):
            self._attach(model=resources.model, sink=self._hook_results, handles=self._handles)
            return True
        else:
            return False

    def _compute(self, resources: HookCallbackResources[ModelTContr]) -> CacheDataTCov:
        _ = resources
        result = self._aggregate(hook_results=self._hook_results)
        self._hook_results.clear()
        self._detach()
        return result

    def _detach(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


HookScoreCallback = HookCallback[ModelTContr, Score, Dict[str, torch.Tensor]]

HookArrayCallback = HookCallback[ModelTContr, Array, Dict[str, torch.Tensor]]


HookPlotCallback = HookCallback[ModelTContr, Plot, Dict[str, torch.Tensor]]


HookScoreCollectionCallback = HookCallback[ModelTContr, ScoreCollection, Dict[str, torch.Tensor]]


HookArrayCollectionCallback = HookCallback[ModelTContr, ArrayCollection, Dict[str, torch.Tensor]]


HookPlotCollectionCallback = HookCallback[ModelTContr, PlotCollection, Dict[str, torch.Tensor]]
