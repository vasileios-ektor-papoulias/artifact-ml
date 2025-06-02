from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict, Generic, TypeVar

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
from artifact_torch.base.model.io import (
    ModelInput,
    ModelOutput,
)

CacheDataT = TypeVar("CacheDataT")
ModelInputT = TypeVar("ModelInputT", bound=ModelInput)
ModelOutputT = TypeVar("ModelOutputT", bound=ModelOutput)


@dataclass
class BatchCallbackResources(PeriodicCallbackResources, Generic[ModelInputT, ModelOutputT]):
    model_input: ModelInputT
    model_output: ModelOutputT


class BatchCallback(
    PeriodicTrackingCallback[
        BatchCallbackResources[ModelInputT, ModelOutputT],
        CacheDataT,
    ],
    Generic[ModelInputT, ModelOutputT, CacheDataT],
):
    def __init__(self, period: int):
        key = self._get_key()
        super().__init__(key=key, period=period)

    @classmethod
    @abstractmethod
    def _get_key(cls) -> str: ...

    @staticmethod
    @abstractmethod
    def _compute_on_batch(model_input: ModelInputT, model_output: ModelOutputT) -> CacheDataT: ...

    @staticmethod
    @abstractmethod
    def _export(key: str, value: CacheDataT, tracking_client: TrackingClient): ...

    def _compute(self, resources: BatchCallbackResources[ModelInputT, ModelOutputT]) -> CacheDataT:
        result = self._compute_on_batch(
            model_input=resources.model_input, model_output=resources.model_output
        )
        return result


class BatchScoreCallback(
    ScoreExportMixin,
    BatchCallback[ModelInputT, ModelOutputT, float],
    Generic[ModelInputT, ModelOutputT],
):
    @classmethod
    @abstractmethod
    def _get_key(cls) -> str: ...

    @staticmethod
    @abstractmethod
    def _compute_on_batch(model_input: ModelInputT, model_output: ModelOutputT) -> float: ...


class BatchArrayCallback(
    ArrayExportMixin,
    BatchCallback[ModelInputT, ModelOutputT, ndarray],
    Generic[ModelInputT, ModelOutputT],
):
    @classmethod
    @abstractmethod
    def _get_key(cls) -> str: ...

    @staticmethod
    @abstractmethod
    def _compute_on_batch(model_input: ModelInputT, model_output: ModelOutputT) -> ndarray: ...


class BatchPlotCallback(
    PlotExportMixin,
    BatchCallback[ModelInputT, ModelOutputT, Figure],
    Generic[ModelInputT, ModelOutputT],
):
    @classmethod
    @abstractmethod
    def _get_key(cls) -> str: ...

    @staticmethod
    @abstractmethod
    def _compute_on_batch(model_input: ModelInputT, model_output: ModelOutputT) -> Figure: ...


class BatchScoreCollectionCallback(
    ScoreCollectionExportMixin,
    BatchCallback[ModelInputT, ModelOutputT, Dict[str, float]],
    Generic[ModelInputT, ModelOutputT],
):
    @classmethod
    @abstractmethod
    def _get_key(cls) -> str: ...

    @staticmethod
    @abstractmethod
    def _compute_on_batch(
        model_input: ModelInputT, model_output: ModelOutputT
    ) -> Dict[str, float]: ...


class BatchArrayCollectionCallback(
    ArrayCollectionExportMixin,
    BatchCallback[ModelInputT, ModelOutputT, Dict[str, ndarray]],
    Generic[ModelInputT, ModelOutputT],
):
    @classmethod
    @abstractmethod
    def _get_key(cls) -> str: ...

    @staticmethod
    @abstractmethod
    def _compute_on_batch(
        model_input: ModelInputT, model_output: ModelOutputT
    ) -> Dict[str, ndarray]: ...


class BatchPlotCollectionCallback(
    PlotCollectionExportMixin,
    BatchCallback[ModelInputT, ModelOutputT, Dict[str, Figure]],
    Generic[ModelInputT, ModelOutputT],
):
    @classmethod
    @abstractmethod
    def _get_key(cls) -> str: ...

    @staticmethod
    @abstractmethod
    def _compute_on_batch(
        model_input: ModelInputT, model_output: ModelOutputT
    ) -> Dict[str, Figure]: ...


class BatchCallbackHandler(
    CacheCallbackHandler[
        BatchCallback[ModelInputT, ModelOutputT, CacheDataT],
        BatchCallbackResources[ModelInputT, ModelOutputT],
        CacheDataT,
    ],
    Generic[ModelInputT, ModelOutputT, CacheDataT],
):
    pass
