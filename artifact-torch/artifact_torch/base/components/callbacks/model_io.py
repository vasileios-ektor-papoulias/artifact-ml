from abc import abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Dict, Generic, List, Optional, Tuple, Type, TypeVar

import torch
import torch.nn as nn
from artifact_experiment.base.callbacks.base import CallbackHandlerSuite
from artifact_experiment.base.callbacks.tracking import (
    ArrayCollectionExportMixin,
    ArrayCollectionHandlerExportMixin,
    ArrayExportMixin,
    ArrayHandlerExportMixin,
    PlotCollectionExportMixin,
    PlotCollectionHandlerExportMixin,
    PlotExportMixin,
    PlotHandlerExportMixin,
    ScoreCollectionExportMixin,
    ScoreCollectionHandlerExportMixin,
    ScoreExportMixin,
    ScoreHandlerExportMixin,
)
from artifact_experiment.base.tracking.client import TrackingClient
from matplotlib.figure import Figure
from numpy import ndarray

from artifact_torch.base.components.callbacks.forward_hook import (
    ForwardHookCallback,
    ForwardHookCallbackHandler,
    ForwardHookCallbackResources,
)
from artifact_torch.base.model.base import Model
from artifact_torch.base.model.io import ModelInput, ModelOutput

ModelOutputTCov = TypeVar("ModelOutputTCov", bound=ModelOutput, covariant=True)


ModelIOCallbackResources = ForwardHookCallbackResources[Model[Any, ModelOutputTCov]]


ModelInputTContr = TypeVar("ModelInputTContr", bound=ModelInput, contravariant=True)
ModelOutputTContr = TypeVar("ModelOutputTContr", bound=ModelOutput, contravariant=True)
CacheDataT = TypeVar("CacheDataT")
BatchResultT = TypeVar("BatchResultT")


class ModelIOCallback(
    ForwardHookCallback[Model[Any, ModelOutputTContr], CacheDataT, BatchResultT],
    Generic[ModelInputTContr, ModelOutputTContr, CacheDataT, BatchResultT],
):
    @classmethod
    @abstractmethod
    def _get_name(cls) -> str: ...

    @classmethod
    @abstractmethod
    def _compute_on_batch(
        cls,
        model_input: ModelInputTContr,
        model_output: ModelOutputTContr,
    ) -> BatchResultT: ...

    @classmethod
    @abstractmethod
    def _aggregate_batch_results(cls, ls_batch_results: List[BatchResultT]) -> CacheDataT: ...

    @staticmethod
    @abstractmethod
    def _export(key: str, value: CacheDataT, tracking_client: TrackingClient): ...

    @classmethod
    def _get_layers(cls, model: Model[ModelInputTContr, ModelOutputTContr]) -> Sequence[nn.Module]:
        return [model]

    @classmethod
    def _hook(
        cls, module: nn.Module, inputs: Tuple[Any, ...], output: Any
    ) -> Optional[BatchResultT]:
        _ = module
        return cls._compute_on_batch(model_input=inputs[0], model_output=output)

    @classmethod
    def _aggregate(cls, hook_results: Dict[str, List[BatchResultT]]) -> CacheDataT:
        ls_batch_results = next(iter(hook_results.values()), [])
        return cls._aggregate_batch_results(ls_batch_results=ls_batch_results)


class ModelIOScoreCallback(
    ScoreExportMixin,
    ModelIOCallback[ModelInputTContr, ModelOutputTContr, float, Dict[str, torch.Tensor]],
):
    @classmethod
    @abstractmethod
    def _get_name(cls) -> str: ...

    @classmethod
    @abstractmethod
    def _compute_on_batch(
        cls, model_input: ModelInputTContr, model_output: ModelOutputTContr
    ) -> Dict[str, torch.Tensor]: ...

    @classmethod
    @abstractmethod
    def _aggregate_batch_results(cls, ls_batch_results: List[Dict[str, torch.Tensor]]) -> float: ...


class ModelIOArrayCallback(
    ArrayExportMixin,
    ModelIOCallback[ModelInputTContr, ModelOutputTContr, ndarray, Dict[str, torch.Tensor]],
):
    @classmethod
    @abstractmethod
    def _get_name(cls) -> str: ...

    @classmethod
    @abstractmethod
    def _compute_on_batch(
        cls, model_input: ModelInputTContr, model_output: ModelOutputTContr
    ) -> Dict[str, torch.Tensor]: ...

    @classmethod
    @abstractmethod
    def _aggregate_batch_results(
        cls, ls_batch_results: List[Dict[str, torch.Tensor]]
    ) -> ndarray: ...


class ModelIOPlotCallback(
    PlotExportMixin,
    ModelIOCallback[ModelInputTContr, ModelOutputTContr, Figure, Dict[str, torch.Tensor]],
):
    @classmethod
    @abstractmethod
    def _get_name(cls) -> str: ...

    @classmethod
    @abstractmethod
    def _compute_on_batch(
        cls, model_input: ModelInputTContr, model_output: ModelOutputTContr
    ) -> Dict[str, torch.Tensor]: ...

    @classmethod
    @abstractmethod
    def _aggregate_batch_results(
        cls, ls_batch_results: List[Dict[str, torch.Tensor]]
    ) -> Figure: ...


class ModelIOScoreCollectionCallback(
    ScoreCollectionExportMixin,
    ModelIOCallback[ModelInputTContr, ModelOutputTContr, Dict[str, float], Dict[str, torch.Tensor]],
):
    @classmethod
    @abstractmethod
    def _get_name(cls) -> str: ...

    @classmethod
    @abstractmethod
    def _compute_on_batch(
        cls, model_input: ModelInputTContr, model_output: ModelOutputTContr
    ) -> Dict[str, torch.Tensor]: ...

    @classmethod
    @abstractmethod
    def _aggregate_batch_results(
        cls, ls_batch_results: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, float]: ...


class ModelIOArrayCollectionCallback(
    ArrayCollectionExportMixin,
    ModelIOCallback[
        ModelInputTContr, ModelOutputTContr, Dict[str, ndarray], Dict[str, torch.Tensor]
    ],
):
    @classmethod
    @abstractmethod
    def _get_name(cls) -> str: ...

    @classmethod
    @abstractmethod
    def _compute_on_batch(
        cls, model_input: ModelInputTContr, model_output: ModelOutputTContr
    ) -> Dict[str, torch.Tensor]: ...

    @classmethod
    @abstractmethod
    def _aggregate_batch_results(
        cls, ls_batch_results: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, ndarray]: ...


class ModelIOPlotCollectionCallback(
    PlotCollectionExportMixin,
    ModelIOCallback[
        ModelInputTContr, ModelOutputTContr, Dict[str, Figure], Dict[str, torch.Tensor]
    ],
):
    @classmethod
    @abstractmethod
    def _get_name(cls) -> str: ...

    @classmethod
    @abstractmethod
    def _compute_on_batch(
        cls, model_input: ModelInputTContr, model_output: ModelOutputTContr
    ) -> Dict[str, torch.Tensor]: ...

    @classmethod
    @abstractmethod
    def _aggregate_batch_results(
        cls, ls_batch_results: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, Figure]: ...


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
