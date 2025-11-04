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

from artifact_torch.base.components.callbacks.forward_hook import (
    ForwardHookArrayCallback,
    ForwardHookArrayCollectionCallback,
    ForwardHookCallback,
    ForwardHookPlotCallback,
    ForwardHookPlotCollectionCallback,
    ForwardHookScoreCallback,
    ForwardHookScoreCollectionCallback,
)
from artifact_torch.base.components.callbacks.hook import HookCallbackResources
from artifact_torch.base.model.base import Model

ForwardHookCallbackTCov = TypeVar(
    "ForwardHookCallbackTCov", bound=ForwardHookCallback, covariant=True
)
ModelTContr = TypeVar("ModelTContr", bound=Model, contravariant=True)
CacheDataT = TypeVar("CacheDataT")


class ForwardHookCallbackHandler(
    TrackingCallbackHandler[
        ForwardHookCallbackTCov,
        HookCallbackResources[ModelTContr],
        CacheDataT,
    ],
    Generic[
        ForwardHookCallbackTCov,
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


class ForwardHookScoreHandler(
    ScoreHandlerExportMixin,
    ForwardHookCallbackHandler[ForwardHookScoreCallback[ModelTContr], ModelTContr, float],
    Generic[ModelTContr],
):
    pass


class ForwardHookArrayHandler(
    ArrayHandlerExportMixin,
    ForwardHookCallbackHandler[ForwardHookArrayCallback[ModelTContr], ModelTContr, ndarray],
    Generic[ModelTContr],
):
    pass


class ForwardHookPlotHandler(
    PlotHandlerExportMixin,
    ForwardHookCallbackHandler[ForwardHookPlotCallback[ModelTContr], ModelTContr, Figure],
    Generic[ModelTContr],
):
    pass


class ForwardHookScoreCollectionHandler(
    ScoreCollectionHandlerExportMixin,
    ForwardHookCallbackHandler[
        ForwardHookScoreCollectionCallback[ModelTContr], ModelTContr, Dict[str, float]
    ],
    Generic[ModelTContr],
):
    pass


class ForwardHookArrayCollectionHandler(
    ArrayCollectionHandlerExportMixin,
    ForwardHookCallbackHandler[
        ForwardHookArrayCollectionCallback[ModelTContr], ModelTContr, Dict[str, ndarray]
    ],
    Generic[ModelTContr],
):
    pass


class ForwardHookPlotCollectionHandler(
    PlotCollectionHandlerExportMixin,
    ForwardHookCallbackHandler[
        ForwardHookPlotCollectionCallback[ModelTContr], ModelTContr, Dict[str, Figure]
    ],
    Generic[ModelTContr],
):
    pass


ForwardHookCallbackHandlerSuiteT = TypeVar(
    "ForwardHookCallbackHandlerSuiteT", bound="ForwardHookCallbackHandlerSuite"
)


@dataclass(frozen=True)
class ForwardHookCallbackHandlerSuite(
    CallbackHandlerSuite[ForwardHookCallbackHandler[Any, ModelTContr, Any]],
    Generic[ModelTContr],
):
    score_handler: ForwardHookScoreHandler[ModelTContr]
    array_handler: ForwardHookArrayHandler[ModelTContr]
    plot_handler: ForwardHookPlotHandler[ModelTContr]
    score_collection_handler: ForwardHookScoreCollectionHandler[ModelTContr]
    array_collection_handler: ForwardHookArrayCollectionHandler[ModelTContr]
    plot_collection_handler: ForwardHookPlotCollectionHandler[ModelTContr]

    @classmethod
    def build(
        cls: Type[ForwardHookCallbackHandlerSuiteT],
        score_callbacks: Sequence[ForwardHookScoreCallback[ModelTContr]],
        array_callbacks: Sequence[ForwardHookArrayCallback[ModelTContr]],
        plot_callbacks: Sequence[ForwardHookPlotCallback[ModelTContr]],
        score_collection_callbacks: Sequence[ForwardHookScoreCollectionCallback[ModelTContr]],
        array_collection_callbacks: Sequence[ForwardHookArrayCollectionCallback[ModelTContr]],
        plot_collection_callbacks: Sequence[ForwardHookPlotCollectionCallback[ModelTContr]],
        tracking_client: Optional[TrackingClient] = None,
    ) -> ForwardHookCallbackHandlerSuiteT:
        handler_suite = cls(
            score_handler=ForwardHookScoreHandler[ModelTContr](
                callbacks=score_callbacks, tracking_client=tracking_client
            ),
            array_handler=ForwardHookArrayHandler[ModelTContr](
                callbacks=array_callbacks, tracking_client=tracking_client
            ),
            plot_handler=ForwardHookPlotHandler[ModelTContr](
                callbacks=plot_callbacks, tracking_client=tracking_client
            ),
            score_collection_handler=ForwardHookScoreCollectionHandler[ModelTContr](
                callbacks=score_collection_callbacks, tracking_client=tracking_client
            ),
            array_collection_handler=ForwardHookArrayCollectionHandler[ModelTContr](
                callbacks=array_collection_callbacks, tracking_client=tracking_client
            ),
            plot_collection_handler=ForwardHookPlotCollectionHandler[ModelTContr](
                callbacks=plot_collection_callbacks, tracking_client=tracking_client
            ),
        )
        return handler_suite
