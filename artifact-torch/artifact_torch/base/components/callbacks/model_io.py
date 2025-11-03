from abc import abstractmethod
from collections.abc import Sequence
from typing import Any, Dict, Generic, List, Optional, Tuple, TypeVar

import torch
import torch.nn as nn
from artifact_experiment.base.callbacks.tracking import (
    ArrayCollectionExportMixin,
    ArrayExportMixin,
    PlotCollectionExportMixin,
    PlotExportMixin,
    ScoreCollectionExportMixin,
    ScoreExportMixin,
)
from artifact_experiment.base.tracking.client import TrackingClient
from matplotlib.figure import Figure
from numpy import ndarray

from artifact_torch.base.components.callbacks.forward_hook import ForwardHookCallback
from artifact_torch.base.components.callbacks.hook import HookCallbackResources
from artifact_torch.base.model.base import Model
from artifact_torch.base.model.io import ModelInput, ModelOutput

ModelOutputTCov = TypeVar("ModelOutputTCov", bound=ModelOutput, covariant=True)


ModelIOCallbackResources = HookCallbackResources[Model[Any, ModelOutputTCov]]


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
