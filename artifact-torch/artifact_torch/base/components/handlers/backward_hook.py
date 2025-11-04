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

from artifact_torch.base.components.callbacks.backward_hook import (
    BackwardHookArrayCallback,
    BackwardHookArrayCollectionCallback,
    BackwardHookCallback,
    BackwardHookPlotCallback,
    BackwardHookPlotCollectionCallback,
    BackwardHookScoreCallback,
    BackwardHookScoreCollectionCallback,
)
from artifact_torch.base.components.callbacks.hook import HookCallbackResources
from artifact_torch.base.model.base import Model

BackwardHookCallbackTCov = TypeVar(
    "BackwardHookCallbackTCov", bound=BackwardHookCallback, covariant=True
)
ModelTContr = TypeVar("ModelTContr", bound=Model, contravariant=True)
CacheDataT = TypeVar("CacheDataT")


class BackwardHookCallbackHandler(
    TrackingCallbackHandler[
        BackwardHookCallbackTCov,
        HookCallbackResources[ModelTContr],
        CacheDataT,
    ],
    Generic[
        BackwardHookCallbackTCov,
        ModelTContr,
        CacheDataT,
    ],
):
    @staticmethod
    @abstractmethod
    def _export(cache: Dict[str, CacheDataT], tracking_client: TrackingClient): ...

    def attach(self, resources: HookCallbackResources[ModelTContr]) -> bool:
        any_attached = False
        for callback in self._ls_callbacks:
            any_attached |= callback.attach(resources=resources)
        return any_attached


class BackwardHookScoreHandler(
    ScoreHandlerExportMixin,
    BackwardHookCallbackHandler[BackwardHookScoreCallback[ModelTContr], ModelTContr, float],
    Generic[ModelTContr],
):
    pass


class BackwardHookArrayHandler(
    ArrayHandlerExportMixin,
    BackwardHookCallbackHandler[BackwardHookArrayCallback[ModelTContr], ModelTContr, ndarray],
    Generic[ModelTContr],
):
    pass


class BackwardHookPlotHandler(
    PlotHandlerExportMixin,
    BackwardHookCallbackHandler[BackwardHookPlotCallback[ModelTContr], ModelTContr, Figure],
    Generic[ModelTContr],
):
    pass


class BackwardHookScoreCollectionHandler(
    ScoreCollectionHandlerExportMixin,
    BackwardHookCallbackHandler[
        BackwardHookScoreCollectionCallback[ModelTContr], ModelTContr, Dict[str, float]
    ],
    Generic[ModelTContr],
):
    pass


class BackwardHookArrayCollectionHandler(
    ArrayCollectionHandlerExportMixin,
    BackwardHookCallbackHandler[
        BackwardHookArrayCollectionCallback[ModelTContr], ModelTContr, Dict[str, ndarray]
    ],
    Generic[ModelTContr],
):
    pass


class BackwardHookPlotCollectionHandler(
    PlotCollectionHandlerExportMixin,
    BackwardHookCallbackHandler[
        BackwardHookPlotCollectionCallback[ModelTContr], ModelTContr, Dict[str, Figure]
    ],
    Generic[ModelTContr],
):
    pass


BackwardHookCallbackHandlerSuiteT = TypeVar(
    "BackwardHookCallbackHandlerSuiteT", bound="BackwardHookCallbackHandlerSuite"
)


@dataclass(frozen=True)
class BackwardHookCallbackHandlerSuite(
    CallbackHandlerSuite[BackwardHookCallbackHandler[Any, ModelTContr, Any]],
    Generic[ModelTContr],
):
    score_handler: BackwardHookScoreHandler[ModelTContr]
    array_handler: BackwardHookArrayHandler[ModelTContr]
    plot_handler: BackwardHookPlotHandler[ModelTContr]
    score_collection_handler: BackwardHookScoreCollectionHandler[ModelTContr]
    array_collection_handler: BackwardHookArrayCollectionHandler[ModelTContr]
    plot_collection_handler: BackwardHookPlotCollectionHandler[ModelTContr]

    @classmethod
    def build(
        cls: Type[BackwardHookCallbackHandlerSuiteT],
        score_callbacks: Sequence[BackwardHookScoreCallback[ModelTContr]],
        array_callbacks: Sequence[BackwardHookArrayCallback[ModelTContr]],
        plot_callbacks: Sequence[BackwardHookPlotCallback[ModelTContr]],
        score_collection_callbacks: Sequence[BackwardHookScoreCollectionCallback[ModelTContr]],
        array_collection_callbacks: Sequence[BackwardHookArrayCollectionCallback[ModelTContr]],
        plot_collection_callbacks: Sequence[BackwardHookPlotCollectionCallback[ModelTContr]],
        tracking_client: Optional[TrackingClient] = None,
    ) -> BackwardHookCallbackHandlerSuiteT:
        handler_suite = cls(
            score_handler=BackwardHookScoreHandler[ModelTContr](
                callbacks=score_callbacks, tracking_client=tracking_client
            ),
            array_handler=BackwardHookArrayHandler[ModelTContr](
                callbacks=array_callbacks, tracking_client=tracking_client
            ),
            plot_handler=BackwardHookPlotHandler[ModelTContr](
                callbacks=plot_callbacks, tracking_client=tracking_client
            ),
            score_collection_handler=BackwardHookScoreCollectionHandler[ModelTContr](
                callbacks=score_collection_callbacks, tracking_client=tracking_client
            ),
            array_collection_handler=BackwardHookArrayCollectionHandler[ModelTContr](
                callbacks=array_collection_callbacks, tracking_client=tracking_client
            ),
            plot_collection_handler=BackwardHookPlotCollectionHandler[ModelTContr](
                callbacks=plot_collection_callbacks, tracking_client=tracking_client
            ),
        )
        return handler_suite
