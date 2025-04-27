from abc import abstractmethod
from typing import Dict, Generic, TypeVar

from matplotlib.figure import Figure
from numpy import ndarray

from artifact_experiment.base.tracking.backend import TrackingBackend
from artifact_experiment.base.tracking.logger import ArtifactLogger

trackingBackendT = TypeVar("trackingBackendT", bound=TrackingBackend)
trackingClientT = TypeVar("trackingClientT", bound="TrackingClient")


class TrackingClient(Generic[trackingBackendT]):
    def __init__(self, backend: trackingBackendT):
        self._backend = backend
        self._score_logger = self._get_score_logger(backend=self._backend)
        self._array_logger = self._get_array_logger(backend=self._backend)
        self._plot_logger = self._get_plot_logger(backend=self._backend)
        self._score_collection_logger = self._get_score_collection_logger(backend=self._backend)
        self._array_collection_logger = self._get_array_collection_logger(backend=self._backend)
        self._plot_collection_logger = self._get_plot_collection_logger(backend=self._backend)

    @property
    def experiment_id(self) -> str:
        return self._backend.experiment_id

    @property
    def backend(self) -> trackingBackendT:
        return self._backend

    @staticmethod
    @abstractmethod
    def _get_score_logger(backend: trackingBackendT) -> ArtifactLogger[float, trackingBackendT]: ...

    @staticmethod
    @abstractmethod
    def _get_array_logger(
        backend: trackingBackendT,
    ) -> ArtifactLogger[ndarray, trackingBackendT]: ...

    @staticmethod
    @abstractmethod
    def _get_plot_logger(backend: trackingBackendT) -> ArtifactLogger[Figure, trackingBackendT]: ...

    @staticmethod
    @abstractmethod
    def _get_score_collection_logger(
        backend: trackingBackendT,
    ) -> ArtifactLogger[Dict[str, float], trackingBackendT]: ...

    @staticmethod
    @abstractmethod
    def _get_array_collection_logger(
        backend: trackingBackendT,
    ) -> ArtifactLogger[Dict[str, ndarray], trackingBackendT]: ...

    @staticmethod
    @abstractmethod
    def _get_plot_collection_logger(
        backend: trackingBackendT,
    ) -> ArtifactLogger[Dict[str, Figure], trackingBackendT]: ...

    def start_experiment(self, experiment_id: str):
        self._backend.start_experiment(experiment_id=experiment_id)

    def stop_experiment(self):
        self._backend.stop_experiment()

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
