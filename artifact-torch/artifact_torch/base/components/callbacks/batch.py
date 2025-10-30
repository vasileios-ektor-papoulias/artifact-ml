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

ModelInputTCov = TypeVar("ModelInputTCov", bound=ModelInput, covariant=True)
ModelOutputTCov = TypeVar("ModelOutputTCov", bound=ModelOutput, covariant=True)


@dataclass
class BatchCallbackResources(PeriodicCallbackResources, Generic[ModelInputTCov, ModelOutputTCov]):
    model_input: ModelInputTCov
    model_output: ModelOutputTCov
    model: Model[ModelInputTCov, ModelOutputTCov]


ModelInputTContr = TypeVar("ModelInputTContr", bound=ModelInput, contravariant=True)
ModelOutputTContr = TypeVar("ModelOutputTContr", bound=ModelOutput, contravariant=True)
CacheDataT = TypeVar("CacheDataT")


class BatchCallback(
    PeriodicTrackingCallback[
        BatchCallbackResources[ModelInputTContr, ModelOutputTContr],
        CacheDataT,
    ],
    Generic[ModelInputTContr, ModelOutputTContr, CacheDataT],
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
        model_input: ModelInputTContr,
        model_output: ModelOutputTContr,
    ) -> CacheDataT: ...

    @staticmethod
    @abstractmethod
    def _export(key: str, value: CacheDataT, tracking_client: TrackingClient): ...

    def _compute(
        self,
        resources: BatchCallbackResources[ModelInputTContr, ModelOutputTContr],
    ) -> CacheDataT:
        result = self._compute_on_batch(
            model_input=resources.model_input,
            model_output=resources.model_output,
        )
        return result


class BatchScoreCallback(
    ScoreExportMixin,
    BatchCallback[ModelInputTContr, ModelOutputTContr, float],
    Generic[ModelInputTContr, ModelOutputTContr],
):
    @classmethod
    @abstractmethod
    def _get_key(cls) -> str: ...

    @staticmethod
    @abstractmethod
    def _compute_on_batch(
        model_input: ModelInputTContr,
        model_output: ModelOutputTContr,
    ) -> float: ...


class BatchArrayCallback(
    ArrayExportMixin,
    BatchCallback[ModelInputTContr, ModelOutputTContr, ndarray],
    Generic[ModelInputTContr, ModelOutputTContr],
):
    @classmethod
    @abstractmethod
    def _get_key(cls) -> str: ...

    @staticmethod
    @abstractmethod
    def _compute_on_batch(
        model_input: ModelInputTContr,
        model_output: ModelOutputTContr,
    ) -> ndarray: ...


class BatchPlotCallback(
    PlotExportMixin,
    BatchCallback[ModelInputTContr, ModelOutputTContr, Figure],
    Generic[ModelInputTContr, ModelOutputTContr],
):
    @classmethod
    @abstractmethod
    def _get_key(cls) -> str: ...

    @staticmethod
    @abstractmethod
    def _compute_on_batch(
        model_input: ModelInputTContr,
        model_output: ModelOutputTContr,
    ) -> Figure: ...


class BatchScoreCollectionCallback(
    ScoreCollectionExportMixin,
    BatchCallback[ModelInputTContr, ModelOutputTContr, Dict[str, float]],
    Generic[ModelInputTContr, ModelOutputTContr],
):
    @classmethod
    @abstractmethod
    def _get_key(cls) -> str: ...

    @staticmethod
    @abstractmethod
    def _compute_on_batch(
        model_input: ModelInputTContr,
        model_output: ModelOutputTContr,
    ) -> Dict[str, float]: ...


class BatchArrayCollectionCallback(
    ArrayCollectionExportMixin,
    BatchCallback[ModelInputTContr, ModelOutputTContr, Dict[str, ndarray]],
    Generic[ModelInputTContr, ModelOutputTContr],
):
    @classmethod
    @abstractmethod
    def _get_key(cls) -> str: ...

    @staticmethod
    @abstractmethod
    def _compute_on_batch(
        model_input: ModelInputTContr,
        model_output: ModelOutputTContr,
    ) -> Dict[str, ndarray]: ...


class BatchPlotCollectionCallback(
    PlotCollectionExportMixin,
    BatchCallback[ModelInputTContr, ModelOutputTContr, Dict[str, Figure]],
    Generic[ModelInputTContr, ModelOutputTContr],
):
    @classmethod
    @abstractmethod
    def _get_key(cls) -> str: ...

    @staticmethod
    @abstractmethod
    def _compute_on_batch(
        model_input: ModelInputTContr,
        model_output: ModelOutputTContr,
    ) -> Dict[str, Figure]: ...


ModelInputT = TypeVar("ModelInputT", bound=ModelInput)
ModelOutputT = TypeVar("ModelOutputT", bound=ModelOutput)


class BatchCallbackHandler(
    CacheCallbackHandler[
        BatchCallback[ModelInputT, ModelOutputT, CacheDataT],
        BatchCallbackResources[ModelInputT, ModelOutputT],
        CacheDataT,
    ],
    Generic[ModelInputT, ModelOutputT, CacheDataT],
):
    pass
