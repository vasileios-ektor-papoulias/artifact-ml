from abc import abstractmethod
from typing import Dict, Generic, Optional, Type, TypeVar

from artifact_experiment.base.experiment_tracking.backend import ExperimentTrackingBackend
from artifact_experiment.base.experiment_tracking.logger import (
    ArrayCollectionLogger,
    ArrayLogger,
    PlotCollectionLogger,
    PlotLogger,
    ScoreCollectionLogger,
    ScoreLogger,
)
from matplotlib.figure import Figure
from numpy import ndarray

trackingBackendT = TypeVar("trackingBackendT", bound=ExperimentTrackingBackend)
trackingClientT = TypeVar("trackingClientT", bound="ExperimentTrackingClient")


class ExperimentTrackingClient(Generic[trackingBackendT]):
    def __init__(self, backend: trackingBackendT):
        self._backend = backend
        self._score_logger = self._get_score_logger(backend=self._backend)
        self._array_logger = self._get_array_logger(backend=self._backend)
        self._plot_logger = self._get_plot_logger(backend=self._backend)
        self._score_collection_logger = self._get_score_collection_logger(backend=self._backend)
        self._array_collection_logger = self._get_array_collection_logger(backend=self._backend)
        self._plot_collection_logger = self._get_plot_collection_logger(backend=self._backend)

    @classmethod
    @abstractmethod
    def build(cls: Type[trackingClientT]) -> trackingClientT: ...

    @property
    def experiment_id(self) -> Optional[str]:
        pass

    @property
    def backend(self) -> trackingBackendT:
        return self._backend

    @staticmethod
    @abstractmethod
    def _get_backend_type() -> Type[trackingBackendT]: ...

    @staticmethod
    @abstractmethod
    def _get_score_logger(backend: trackingBackendT) -> ScoreLogger[trackingBackendT]: ...

    @staticmethod
    @abstractmethod
    def _get_array_logger(
        backend: trackingBackendT,
    ) -> ArrayLogger[trackingBackendT]: ...

    @staticmethod
    @abstractmethod
    def _get_plot_logger(backend: trackingBackendT) -> PlotLogger[trackingBackendT]: ...

    @staticmethod
    @abstractmethod
    def _get_score_collection_logger(
        backend: trackingBackendT,
    ) -> ScoreCollectionLogger[trackingBackendT]: ...

    @staticmethod
    @abstractmethod
    def _get_array_collection_logger(
        backend: trackingBackendT,
    ) -> ArrayCollectionLogger[trackingBackendT]: ...

    @staticmethod
    @abstractmethod
    def _get_plot_collection_logger(
        backend: trackingBackendT,
    ) -> PlotCollectionLogger[trackingBackendT]: ...

    def start_experiment(self, experiment_id: str):
        self._backend.start(experiment_id=experiment_id)

    def stop_experiment(self):
        self._backend.stop()

    def log_score(self, score: float, name: str):
        self._score_logger.log(artifact=score, name=name)

    def log_array(self, array: ndarray, name: str):
        self._array_logger.log(artifact=array, name=name)

    def log_plot(self, plot: Figure, name: str):
        self._plot_logger.log(artifact=plot, name=name)

    def log_score_collection(self, score_collection: Dict[str, float], name: str):
        self._score_collection_logger.log(artifact=score_collection, name=name)

    def log_array_collection(self, array_collection: Dict[str, ndarray], name: str):
        self._array_collection_logger.log(artifact=array_collection, name=name)

    def log_plot_collection(self, plot_collection: Dict[str, Figure], name: str):
        self._plot_collection_logger.log(artifact=plot_collection, name=name)
