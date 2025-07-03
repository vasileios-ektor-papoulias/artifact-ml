from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict, Generic, Optional, TypeVar

from artifact_experiment.base.callbacks.tracking import (
    ArrayCollectionExportMixin,
    ArrayExportMixin,
    CacheCallbackHandler,
    PlotCollectionExportMixin,
    PlotExportMixin,
    ScoreCollectionExportMixin,
    ScoreExportMixin,
)
from artifact_experiment.base.tracking.client import TrackingClient
from matplotlib.figure import Figure
from numpy import ndarray

from artifact_torch.base.components.callbacks.periodic import (
    PeriodicCallbackResources,
    PeriodicTrackingCallback,
)
from artifact_torch.base.model.base import Model
from artifact_torch.base.model.io import (
    ModelInput,
    ModelOutput,
)

CacheDataT = TypeVar("CacheDataT")
ModelInputTCov = TypeVar("ModelInputTCov", bound=ModelInput, covariant=True)
ModelOutputTCov = TypeVar("ModelOutputTCov", bound=ModelOutput, covariant=True)
ModelTCov = TypeVar("ModelTCov", bound=Model, covariant=True)


@dataclass
class BatchCallbackResources(
    PeriodicCallbackResources, Generic[ModelInputTCov, ModelOutputTCov, ModelTCov]
):
    model_input: ModelInputTCov
    model_output: ModelOutputTCov
    model: ModelTCov


ModelInputTContr = TypeVar("ModelInputTContr", bound=ModelInput, contravariant=True)
ModelOutputTContr = TypeVar("ModelOutputTContr", bound=ModelOutput, contravariant=True)
ModelTContr = TypeVar("ModelTContr", bound=Model, contravariant=True)


class BatchCallback(
    PeriodicTrackingCallback[
        BatchCallbackResources[ModelInputTContr, ModelOutputTContr, ModelTContr],
        CacheDataT,
    ],
    Generic[ModelInputTContr, ModelOutputTContr, ModelTContr, CacheDataT],
):
    def __init__(self, period: int, tracking_client: Optional[TrackingClient] = None):
        key = self._get_key()
        super().__init__(key=key, period=period, tracking_client=tracking_client)

    @classmethod
    @abstractmethod
    def _get_key(cls) -> str: ...

    @staticmethod
    @abstractmethod
    def _compute_on_batch(
        model_input: ModelInputTContr, model_output: ModelOutputTContr, model: ModelTContr
    ) -> CacheDataT: ...

    @staticmethod
    @abstractmethod
    def _export(key: str, value: CacheDataT, tracking_client: TrackingClient): ...

    def _compute(
        self, resources: BatchCallbackResources[ModelInputTContr, ModelOutputTContr, ModelTContr]
    ) -> CacheDataT:
        result = self._compute_on_batch(
            model_input=resources.model_input,
            model_output=resources.model_output,
            model=resources.model,
        )
        return result


class BatchScoreCallback(
    ScoreExportMixin,
    BatchCallback[ModelInputTContr, ModelOutputTContr, ModelTContr, float],
    Generic[ModelInputTContr, ModelOutputTContr, ModelTContr],
):
    @classmethod
    @abstractmethod
    def _get_key(cls) -> str: ...

    @staticmethod
    @abstractmethod
    def _compute_on_batch(
        model_input: ModelInputTContr, model_output: ModelOutputTContr, model: ModelTContr
    ) -> float: ...


class BatchArrayCallback(
    ArrayExportMixin,
    BatchCallback[ModelInputTContr, ModelOutputTContr, ModelTContr, ndarray],
    Generic[ModelInputTContr, ModelOutputTContr, ModelTContr],
):
    @classmethod
    @abstractmethod
    def _get_key(cls) -> str: ...

    @staticmethod
    @abstractmethod
    def _compute_on_batch(
        model_input: ModelInputTContr, model_output: ModelOutputTContr, model: ModelTContr
    ) -> ndarray: ...


class BatchPlotCallback(
    PlotExportMixin,
    BatchCallback[ModelInputTContr, ModelOutputTContr, ModelTContr, Figure],
    Generic[ModelInputTContr, ModelOutputTContr, ModelTContr],
):
    @classmethod
    @abstractmethod
    def _get_key(cls) -> str: ...

    @staticmethod
    @abstractmethod
    def _compute_on_batch(
        model_input: ModelInputTContr, model_output: ModelOutputTContr, model: ModelTContr
    ) -> Figure: ...


class BatchScoreCollectionCallback(
    ScoreCollectionExportMixin,
    BatchCallback[ModelInputTContr, ModelOutputTContr, ModelTContr, Dict[str, float]],
    Generic[ModelInputTContr, ModelOutputTContr, ModelTContr],
):
    @classmethod
    @abstractmethod
    def _get_key(cls) -> str: ...

    @staticmethod
    @abstractmethod
    def _compute_on_batch(
        model_input: ModelInputTContr, model_output: ModelOutputTContr, model: ModelTContr
    ) -> Dict[str, float]: ...


class BatchArrayCollectionCallback(
    ArrayCollectionExportMixin,
    BatchCallback[ModelInputTContr, ModelOutputTContr, ModelTContr, Dict[str, ndarray]],
    Generic[ModelInputTContr, ModelOutputTContr, ModelTContr],
):
    @classmethod
    @abstractmethod
    def _get_key(cls) -> str: ...

    @staticmethod
    @abstractmethod
    def _compute_on_batch(
        model_input: ModelInputTContr, model_output: ModelOutputTContr, model: ModelTContr
    ) -> Dict[str, ndarray]: ...


class BatchPlotCollectionCallback(
    PlotCollectionExportMixin,
    BatchCallback[ModelInputTContr, ModelOutputTContr, ModelTContr, Dict[str, Figure]],
    Generic[ModelInputTContr, ModelOutputTContr, ModelTContr],
):
    @classmethod
    @abstractmethod
    def _get_key(cls) -> str: ...

    @staticmethod
    @abstractmethod
    def _compute_on_batch(
        model_input: ModelInputTContr, model_output: ModelOutputTContr, model: ModelTContr
    ) -> Dict[str, Figure]: ...


ModelInputT = TypeVar("ModelInputT", bound=ModelInput)
ModelOutputT = TypeVar("ModelOutputT", bound=ModelOutput)
ModelT = TypeVar("ModelT", bound=Model)


class BatchCallbackHandler(
    CacheCallbackHandler[
        BatchCallback[ModelInputT, ModelOutputT, ModelT, CacheDataT],
        BatchCallbackResources[ModelInputT, ModelOutputT, ModelT],
        CacheDataT,
    ],
    Generic[ModelInputT, ModelOutputT, ModelT, CacheDataT],
):
    pass
