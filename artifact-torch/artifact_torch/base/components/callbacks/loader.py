from abc import abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Dict, Generic, List, Optional, Tuple, Type, TypeVar

import torch
import torch.nn as nn
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
)
from artifact_torch.base.model.base import Model
from artifact_torch.base.model.io import ModelInput, ModelOutput

ModelInputTContr = TypeVar("ModelInputTContr", bound=ModelInput, contravariant=True)
ModelOutputTContr = TypeVar("ModelOutputTContr", bound=ModelOutput, contravariant=True)
CacheDataT = TypeVar("CacheDataT")
BatchResultT = TypeVar("BatchResultT")


class DataLoaderCallback(
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


class DataLoaderScore(
    ScoreExportMixin,
    DataLoaderCallback[ModelInputTContr, ModelOutputTContr, float, Dict[str, torch.Tensor]],
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


class DataLoaderArray(
    ArrayExportMixin,
    DataLoaderCallback[ModelInputTContr, ModelOutputTContr, ndarray, Dict[str, torch.Tensor]],
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


class DataLoaderPlot(
    PlotExportMixin,
    DataLoaderCallback[ModelInputTContr, ModelOutputTContr, Figure, Dict[str, torch.Tensor]],
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


class DataLoaderScoreCollection(
    ScoreCollectionExportMixin,
    DataLoaderCallback[
        ModelInputTContr, ModelOutputTContr, Dict[str, float], Dict[str, torch.Tensor]
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
    ) -> Dict[str, float]: ...


class DataLoaderArrayCollection(
    ArrayCollectionExportMixin,
    DataLoaderCallback[
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


class DataLoaderPlotCollection(
    PlotCollectionExportMixin,
    DataLoaderCallback[
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


ModelInputT = TypeVar("ModelInputT", bound=ModelInput)
ModelOutputT = TypeVar("ModelOutputT", bound=ModelOutput)
DataLoaderCallbackT = TypeVar("DataLoaderCallbackT", bound=DataLoaderCallback)


class DataLoaderCallbackHandler(
    ForwardHookCallbackHandler[
        DataLoaderCallbackT,
        Model[ModelInputT, ModelOutputT],
        CacheDataT,
    ],
    Generic[DataLoaderCallbackT, ModelInputT, ModelOutputT, CacheDataT],
):
    @staticmethod
    @abstractmethod
    def _export(cache: Dict[str, CacheDataT], tracking_client: TrackingClient): ...


class DataLoaderScoreHandler(
    ScoreHandlerExportMixin,
    DataLoaderCallbackHandler[
        DataLoaderScore[ModelInputT, ModelOutputT],
        ModelInputT,
        ModelOutputT,
        float,
    ],
    Generic[ModelInputT, ModelOutputT],
): ...


class DataLoaderArrayHandler(
    ArrayHandlerExportMixin,
    DataLoaderCallbackHandler[
        DataLoaderArray[ModelInputT, ModelOutputT],
        ModelInputT,
        ModelOutputT,
        ndarray,
    ],
    Generic[ModelInputT, ModelOutputT],
): ...


class DataLoaderPlotHandler(
    PlotHandlerExportMixin,
    DataLoaderCallbackHandler[
        DataLoaderPlot[ModelInputT, ModelOutputT],
        ModelInputT,
        ModelOutputT,
        Figure,
    ],
    Generic[ModelInputT, ModelOutputT],
): ...


class DataLoaderScoreCollectionHandler(
    ScoreCollectionHandlerExportMixin,
    DataLoaderCallbackHandler[
        DataLoaderScoreCollection[ModelInputT, ModelOutputT],
        ModelInputT,
        ModelOutputT,
        Dict[str, float],
    ],
    Generic[ModelInputT, ModelOutputT],
): ...


class DataLoaderArrayCollectionHandler(
    ArrayCollectionHandlerExportMixin,
    DataLoaderCallbackHandler[
        DataLoaderArrayCollection[ModelInputT, ModelOutputT],
        ModelInputT,
        ModelOutputT,
        Dict[str, ndarray],
    ],
    Generic[ModelInputT, ModelOutputT],
): ...


class DataLoaderPlotCollectionHandler(
    PlotCollectionHandlerExportMixin,
    DataLoaderCallbackHandler[
        DataLoaderPlotCollection[ModelInputT, ModelOutputT],
        ModelInputT,
        ModelOutputT,
        Dict[str, Figure],
    ],
    Generic[ModelInputT, ModelOutputT],
): ...


DataLoaderHandlersT = TypeVar("DataLoaderHandlersT", bound="DataLoaderHandlers")


@dataclass
class DataLoaderHandlers(Generic[ModelInputT, ModelOutputT]):
    score_handler: DataLoaderScoreHandler[ModelInputT, ModelOutputT]
    array_handler: DataLoaderArrayHandler[ModelInputT, ModelOutputT]
    plot_handler: DataLoaderPlotHandler[ModelInputT, ModelOutputT]
    score_collection_handler: DataLoaderScoreCollectionHandler[ModelInputT, ModelOutputT]
    array_collection_handler: DataLoaderArrayCollectionHandler[ModelInputT, ModelOutputT]
    plot_collection_handler: DataLoaderPlotCollectionHandler[ModelInputT, ModelOutputT]

    @classmethod
    def build(
        cls: Type[DataLoaderHandlersT],
        ls_score_callbacks: List[DataLoaderScore[ModelInputT, ModelOutputT]],
        ls_array_callbacks: List[DataLoaderArray[ModelInputT, ModelOutputT]],
        ls_plot_callbacks: List[DataLoaderPlot[ModelInputT, ModelOutputT]],
        ls_score_collection_callbacks: List[DataLoaderScoreCollection[ModelInputT, ModelOutputT]],
        ls_array_collection_callbacks: List[DataLoaderArrayCollection[ModelInputT, ModelOutputT]],
        ls_plot_collection_callbacks: List[DataLoaderPlotCollection[ModelInputT, ModelOutputT]],
        tracking_client: Optional[TrackingClient] = None,
    ) -> DataLoaderHandlersT:
        handlers = cls(
            score_handler=DataLoaderScoreHandler[ModelInputT, ModelOutputT](
                ls_callbacks=ls_score_callbacks, tracking_client=tracking_client
            ),
            array_handler=DataLoaderArrayHandler[ModelInputT, ModelOutputT](
                ls_callbacks=ls_array_callbacks, tracking_client=tracking_client
            ),
            plot_handler=DataLoaderPlotHandler[ModelInputT, ModelOutputT](
                ls_callbacks=ls_plot_callbacks, tracking_client=tracking_client
            ),
            score_collection_handler=DataLoaderScoreCollectionHandler[ModelInputT, ModelOutputT](
                ls_callbacks=ls_score_collection_callbacks, tracking_client=tracking_client
            ),
            array_collection_handler=DataLoaderArrayCollectionHandler[ModelInputT, ModelOutputT](
                ls_callbacks=ls_array_collection_callbacks, tracking_client=tracking_client
            ),
            plot_collection_handler=DataLoaderPlotCollectionHandler[ModelInputT, ModelOutputT](
                ls_callbacks=ls_plot_collection_callbacks, tracking_client=tracking_client
            ),
        )
        return handlers
