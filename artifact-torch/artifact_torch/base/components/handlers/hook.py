from typing import Any, Dict, Generic, TypeVar

from artifact_core._base.artifact_dependencies import ArtifactResult
from artifact_experiment.base.components.handlers.tracking import TrackingCallbackHandler
from matplotlib.figure import Figure
from numpy import ndarray

from artifact_torch.base.components.callbacks.hook import HookCallback, HookCallbackResources
from artifact_torch.base.model.base import Model

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


HookScoreHandler = HookCallbackHandler[ModelTContr, float]


HookArrayHandler = HookCallbackHandler[ModelTContr, ndarray]


HookPlotHandler = HookCallbackHandler[ModelTContr, Figure]


HookScoreCollectionHandler = HookCallbackHandler[ModelTContr, Dict[str, float]]


HookArrayCollectionHandler = HookCallbackHandler[ModelTContr, Dict[str, ndarray]]


HookPlotCollectionHandler = HookCallbackHandler[ModelTContr, Dict[str, Figure]]
