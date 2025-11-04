from abc import abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Dict, Generic, Optional, Type, TypeVar

from artifact_experiment.base.handlers.base import CallbackHandlerSuite
from artifact_experiment.base.handlers.tracking import (
    ArrayCollectionHandlerExportMixin,
    ArrayHandlerExportMixin,
    PlotCollectionHandlerExportMixin,
    PlotHandlerExportMixin,
    ScoreCollectionHandlerExportMixin,
    ScoreHandlerExportMixin,
    TrackingCallbackHandler,
)
from artifact_experiment.base.tracking.client import TrackingClient
from matplotlib.figure import Figure
from numpy import ndarray

from artifact_torch.base.components.callbacks.hook import HookCallback, HookCallbackResources
from artifact_torch.base.model.base import Model

ModelTContr = TypeVar("ModelTContr", bound=Model, contravariant=True)
CacheDataT = TypeVar("CacheDataT")


class HookCallbackHandler(
    TrackingCallbackHandler[
        HookCallback[ModelTContr, CacheDataT, Any],
        HookCallbackResources[ModelTContr],
        CacheDataT,
    ],
    Generic[ModelTContr, CacheDataT],
):
    @staticmethod
    @abstractmethod
    def _export(cache: Dict[str, CacheDataT], tracking_client: TrackingClient): ...

    def attach(self, resources: HookCallbackResources[ModelTContr]) -> bool:
        any_attached = False
        for callback in self._ls_callbacks:
            any_attached |= callback.attach(resources=resources)
        return any_attached


class HookScoreHandler(
    ScoreHandlerExportMixin,
    HookCallbackHandler[ModelTContr, float],
    Generic[ModelTContr],
):
    pass


class HookArrayHandler(
    ArrayHandlerExportMixin,
    HookCallbackHandler[ModelTContr, ndarray],
    Generic[ModelTContr],
):
    pass


class HookPlotHandler(
    PlotHandlerExportMixin,
    HookCallbackHandler[ModelTContr, Figure],
    Generic[ModelTContr],
):
    pass


class HookScoreCollectionHandler(
    ScoreCollectionHandlerExportMixin,
    HookCallbackHandler[ModelTContr, Dict[str, float]],
    Generic[ModelTContr],
):
    pass


class HookArrayCollectionHandler(
    ArrayCollectionHandlerExportMixin,
    HookCallbackHandler[ModelTContr, Dict[str, ndarray]],
    Generic[ModelTContr],
):
    pass


class HookPlotCollectionHandler(
    PlotCollectionHandlerExportMixin,
    HookCallbackHandler[ModelTContr, Dict[str, Figure]],
    Generic[ModelTContr],
):
    pass


HookCallbackHandlerSuiteT = TypeVar("HookCallbackHandlerSuiteT", bound="HookCallbackHandlerSuite")


@dataclass(frozen=True)
class HookCallbackHandlerSuite(
    CallbackHandlerSuite[HookCallbackHandler[ModelTContr, Any]],
    Generic[ModelTContr],
):
    score_handler: HookScoreHandler[ModelTContr]
    array_handler: HookArrayHandler[ModelTContr]
    plot_handler: HookPlotHandler[ModelTContr]
    score_collection_handler: HookScoreCollectionHandler[ModelTContr]
    array_collection_handler: HookArrayCollectionHandler[ModelTContr]
    plot_collection_handler: HookPlotCollectionHandler[ModelTContr]

    @classmethod
    def build(
        cls: Type[HookCallbackHandlerSuiteT],
        score_callbacks: Sequence[HookCallback[ModelTContr, float, Any]],
        array_callbacks: Sequence[HookCallback[ModelTContr, ndarray, Any]],
        plot_callbacks: Sequence[HookCallback[ModelTContr, Figure, Any]],
        score_collection_callbacks: Sequence[HookCallback[ModelTContr, Dict[str, float], Any]],
        array_collection_callbacks: Sequence[HookCallback[ModelTContr, Dict[str, ndarray], Any]],
        plot_collection_callbacks: Sequence[HookCallback[ModelTContr, Dict[str, Figure], Any]],
        tracking_client: Optional[TrackingClient] = None,
    ) -> HookCallbackHandlerSuiteT:
        handler_suite = cls(
            score_handler=HookScoreHandler[ModelTContr](
                callbacks=score_callbacks, tracking_client=tracking_client
            ),
            array_handler=HookArrayHandler[ModelTContr](
                callbacks=array_callbacks, tracking_client=tracking_client
            ),
            plot_handler=HookPlotHandler[ModelTContr](
                callbacks=plot_callbacks, tracking_client=tracking_client
            ),
            score_collection_handler=HookScoreCollectionHandler[ModelTContr](
                callbacks=score_collection_callbacks, tracking_client=tracking_client
            ),
            array_collection_handler=HookArrayCollectionHandler[ModelTContr](
                callbacks=array_collection_callbacks, tracking_client=tracking_client
            ),
            plot_collection_handler=HookPlotCollectionHandler[ModelTContr](
                callbacks=plot_collection_callbacks, tracking_client=tracking_client
            ),
        )
        return handler_suite
