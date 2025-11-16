from typing import Any, Generic, TypeVar

from artifact_core.typing import (
    Array,
    ArrayCollection,
    ArtifactResult,
    Plot,
    PlotCollection,
    Score,
    ScoreCollection,
)
from artifact_experiment.spi.handlers import TrackingCallbackHandler

from artifact_torch._base.components.callbacks.hook import HookCallback, HookCallbackResources
from artifact_torch._base.model.base import Model

ModelTContr = TypeVar("ModelTContr", bound=Model, contravariant=True)
CacheDataTCov = TypeVar("CacheDataTCov", bound=ArtifactResult, covariant=True)


class HookCallbackHandler(
    TrackingCallbackHandler[
        HookCallback[ModelTContr, CacheDataTCov, Any],
        HookCallbackResources[ModelTContr],
        CacheDataTCov,
    ],
    Generic[ModelTContr, CacheDataTCov],
):
    def attach(self, resources: HookCallbackResources[ModelTContr]) -> bool:
        any_attached = False
        for callback in self._ls_callbacks:
            any_attached |= callback.attach(resources=resources)
        return any_attached


HookScoreHandler = HookCallbackHandler[ModelTContr, Score]


HookArrayHandler = HookCallbackHandler[ModelTContr, Array]


HookPlotHandler = HookCallbackHandler[ModelTContr, Plot]


HookScoreCollectionHandler = HookCallbackHandler[ModelTContr, ScoreCollection]


HookArrayCollectionHandler = HookCallbackHandler[ModelTContr, ArrayCollection]


HookPlotCollectionHandler = HookCallbackHandler[ModelTContr, PlotCollection]
