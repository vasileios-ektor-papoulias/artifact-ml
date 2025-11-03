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
)
from artifact_experiment.base.tracking.client import TrackingClient
from matplotlib.figure import Figure
from numpy import ndarray

from artifact_torch.base.components.callbacks.model_io import (
    ModelIOArrayCallback,
    ModelIOArrayCollectionCallback,
    ModelIOCallback,
    ModelIOPlotCallback,
    ModelIOPlotCollectionCallback,
    ModelIOScoreCallback,
    ModelIOScoreCollectionCallback,
)
from artifact_torch.base.components.handlers.forward_hook import ForwardHookCallbackHandler
from artifact_torch.base.model.base import Model
from artifact_torch.base.model.io import ModelInput, ModelOutput

ModelInputTContr = TypeVar("ModelInputTContr", bound=ModelInput, contravariant=True)
ModelOutputTContr = TypeVar("ModelOutputTContr", bound=ModelOutput, contravariant=True)
CacheDataT = TypeVar("CacheDataT")


class ModelIOCallbackHandler(
    ForwardHookCallbackHandler[
        ModelIOCallback[ModelInputTContr, ModelOutputTContr, CacheDataT, Any],
        Model[Any, ModelOutputTContr],
        CacheDataT,
    ],
    Generic[ModelInputTContr, ModelOutputTContr, CacheDataT],
):
    @staticmethod
    @abstractmethod
    def _export(cache: Dict[str, CacheDataT], tracking_client: TrackingClient): ...


class ModelIOScoreHandler(
    ScoreHandlerExportMixin,
    ModelIOCallbackHandler[ModelInputTContr, ModelOutputTContr, float],
    Generic[ModelInputTContr, ModelOutputTContr],
): ...


class ModelIOArrayHandler(
    ArrayHandlerExportMixin,
    ModelIOCallbackHandler[ModelInputTContr, ModelOutputTContr, ndarray],
    Generic[ModelInputTContr, ModelOutputTContr],
): ...


class ModelIOPlotHandler(
    PlotHandlerExportMixin,
    ModelIOCallbackHandler[ModelInputTContr, ModelOutputTContr, Figure],
    Generic[ModelInputTContr, ModelOutputTContr],
): ...


class ModelIOScoreCollectionHandler(
    ScoreCollectionHandlerExportMixin,
    ModelIOCallbackHandler[ModelInputTContr, ModelOutputTContr, Dict[str, float]],
    Generic[ModelInputTContr, ModelOutputTContr],
): ...


class ModelIOArrayCollectionHandler(
    ArrayCollectionHandlerExportMixin,
    ModelIOCallbackHandler[ModelInputTContr, ModelOutputTContr, Dict[str, ndarray]],
    Generic[ModelInputTContr, ModelOutputTContr],
): ...


class ModelIOPlotCollectionHandler(
    PlotCollectionHandlerExportMixin,
    ModelIOCallbackHandler[ModelInputTContr, ModelOutputTContr, Dict[str, Figure]],
    Generic[ModelInputTContr, ModelOutputTContr],
): ...


ModelIOHandlerSuiteT = TypeVar("ModelIOHandlerSuiteT", bound="ModelIOHandlerSuite")


@dataclass(frozen=True)
class ModelIOHandlerSuite(
    CallbackHandlerSuite[ModelIOCallbackHandler[ModelInputTContr, ModelOutputTContr, Any]],
    Generic[ModelInputTContr, ModelOutputTContr],
):
    score_handler: ModelIOScoreHandler[ModelInputTContr, ModelOutputTContr]
    array_handler: ModelIOArrayHandler[ModelInputTContr, ModelOutputTContr]
    plot_handler: ModelIOPlotHandler[ModelInputTContr, ModelOutputTContr]
    score_collection_handler: ModelIOScoreCollectionHandler[ModelInputTContr, ModelOutputTContr]
    array_collection_handler: ModelIOArrayCollectionHandler[ModelInputTContr, ModelOutputTContr]
    plot_collection_handler: ModelIOPlotCollectionHandler[ModelInputTContr, ModelOutputTContr]

    @classmethod
    def build(
        cls: Type[ModelIOHandlerSuiteT],
        score_callbacks: Sequence[ModelIOScoreCallback[ModelInputTContr, ModelOutputTContr]],
        array_callbacks: Sequence[ModelIOArrayCallback[ModelInputTContr, ModelOutputTContr]],
        plot_callbacks: Sequence[ModelIOPlotCallback[ModelInputTContr, ModelOutputTContr]],
        score_collection_callbacks: Sequence[
            ModelIOScoreCollectionCallback[ModelInputTContr, ModelOutputTContr]
        ],
        array_collection_callbacks: Sequence[
            ModelIOArrayCollectionCallback[ModelInputTContr, ModelOutputTContr]
        ],
        plot_collection_callbacks: Sequence[
            ModelIOPlotCollectionCallback[ModelInputTContr, ModelOutputTContr]
        ],
        tracking_client: Optional[TrackingClient] = None,
    ) -> ModelIOHandlerSuiteT:
        handlers = cls(
            score_handler=ModelIOScoreHandler[ModelInputTContr, ModelOutputTContr](
                callbacks=score_callbacks, tracking_client=tracking_client
            ),
            array_handler=ModelIOArrayHandler[ModelInputTContr, ModelOutputTContr](
                callbacks=array_callbacks, tracking_client=tracking_client
            ),
            plot_handler=ModelIOPlotHandler[ModelInputTContr, ModelOutputTContr](
                callbacks=plot_callbacks, tracking_client=tracking_client
            ),
            score_collection_handler=ModelIOScoreCollectionHandler[
                ModelInputTContr, ModelOutputTContr
            ](callbacks=score_collection_callbacks, tracking_client=tracking_client),
            array_collection_handler=ModelIOArrayCollectionHandler[
                ModelInputTContr, ModelOutputTContr
            ](callbacks=array_collection_callbacks, tracking_client=tracking_client),
            plot_collection_handler=ModelIOPlotCollectionHandler[
                ModelInputTContr, ModelOutputTContr
            ](callbacks=plot_collection_callbacks, tracking_client=tracking_client),
        )
        return handlers
