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
    ModelIOPlotCallback,
    ModelIOPlotCollectionCallback,
    ModelIOScoreCallback,
    ModelIOScoreCollectionCallback,
)
from artifact_torch.base.components.handlers.hook import HookCallbackHandler
from artifact_torch.base.model.base import Model
from artifact_torch.base.model.io import ModelInput, ModelOutput

ModelInputTContr = TypeVar("ModelInputTContr", bound=ModelInput, contravariant=True)
ModelOutputTContr = TypeVar("ModelOutputTContr", bound=ModelOutput, contravariant=True)
CacheDataT = TypeVar("CacheDataT")


class ModelIOCallbackHandler(
    HookCallbackHandler[Model[Any, ModelOutputTContr], CacheDataT],
    Generic[ModelOutputTContr, CacheDataT],
):
    @staticmethod
    @abstractmethod
    def _export(cache: Dict[str, CacheDataT], tracking_client: TrackingClient): ...


class ModelIOScoreHandler(
    ScoreHandlerExportMixin,
    ModelIOCallbackHandler[ModelOutputTContr, float],
    Generic[ModelOutputTContr],
): ...


class ModelIOArrayHandler(
    ArrayHandlerExportMixin,
    ModelIOCallbackHandler[ModelOutputTContr, ndarray],
    Generic[ModelOutputTContr],
): ...


class ModelIOPlotHandler(
    PlotHandlerExportMixin,
    ModelIOCallbackHandler[ModelOutputTContr, Figure],
    Generic[ModelOutputTContr],
): ...


class ModelIOScoreCollectionHandler(
    ScoreCollectionHandlerExportMixin,
    ModelIOCallbackHandler[ModelOutputTContr, Dict[str, float]],
    Generic[ModelOutputTContr],
): ...


class ModelIOArrayCollectionHandler(
    ArrayCollectionHandlerExportMixin,
    ModelIOCallbackHandler[ModelOutputTContr, Dict[str, ndarray]],
    Generic[ModelOutputTContr],
): ...


class ModelIOPlotCollectionHandler(
    PlotCollectionHandlerExportMixin,
    ModelIOCallbackHandler[ModelOutputTContr, Dict[str, Figure]],
    Generic[ModelOutputTContr],
): ...


ModelIOCallbackHandlerSuiteT = TypeVar(
    "ModelIOCallbackHandlerSuiteT", bound="ModelIOCallbackHandlerSuite"
)


@dataclass(frozen=True)
class ModelIOCallbackHandlerSuite(
    CallbackHandlerSuite[ModelIOCallbackHandler[ModelOutputTContr, Any]],
    Generic[ModelInputTContr, ModelOutputTContr],
):
    score_handler: ModelIOScoreHandler[ModelOutputTContr]
    array_handler: ModelIOArrayHandler[ModelOutputTContr]
    plot_handler: ModelIOPlotHandler[ModelOutputTContr]
    score_collection_handler: ModelIOScoreCollectionHandler[ModelOutputTContr]
    array_collection_handler: ModelIOArrayCollectionHandler[ModelOutputTContr]
    plot_collection_handler: ModelIOPlotCollectionHandler[ModelOutputTContr]

    @classmethod
    def build(
        cls: Type[ModelIOCallbackHandlerSuiteT],
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
    ) -> ModelIOCallbackHandlerSuiteT:
        handler_suite = cls(
            score_handler=ModelIOScoreHandler[ModelOutputTContr](
                callbacks=score_callbacks, tracking_client=tracking_client
            ),
            array_handler=ModelIOArrayHandler[ModelOutputTContr](
                callbacks=array_callbacks, tracking_client=tracking_client
            ),
            plot_handler=ModelIOPlotHandler[ModelOutputTContr](
                callbacks=plot_callbacks, tracking_client=tracking_client
            ),
            score_collection_handler=ModelIOScoreCollectionHandler[ModelOutputTContr](
                callbacks=score_collection_callbacks, tracking_client=tracking_client
            ),
            array_collection_handler=ModelIOArrayCollectionHandler[ModelOutputTContr](
                callbacks=array_collection_callbacks, tracking_client=tracking_client
            ),
            plot_collection_handler=ModelIOPlotCollectionHandler[ModelOutputTContr](
                callbacks=plot_collection_callbacks, tracking_client=tracking_client
            ),
        )
        return handler_suite
