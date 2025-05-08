from abc import abstractmethod
from typing import Dict, Generic, TypeVar

from matplotlib.figure import Figure
from numpy import ndarray

from artifact_experiment.base.tracking.adapter import RunAdapter
from artifact_experiment.base.tracking.logger import ArtifactLogger

runAdapterT = TypeVar("runAdapterT", bound=RunAdapter)
trackingClientT = TypeVar("trackingClientT", bound="TrackingClient")


class TrackingClient(Generic[runAdapterT]):
    def __init__(self, run: runAdapterT):
        self._run = run
        self._score_logger = self._get_score_logger(run=self._run)
        self._array_logger = self._get_array_logger(run=self._run)
        self._plot_logger = self._get_plot_logger(run=self._run)
        self._score_collection_logger = self._get_score_collection_logger(run=self._run)
        self._array_collection_logger = self._get_array_collection_logger(run=self._run)
        self._plot_collection_logger = self._get_plot_collection_logger(run=self._run)

    @property
    def run(self) -> runAdapterT:
        return self._run

    @run.setter
    def run(self, run: runAdapterT):
        self._run = run

    @staticmethod
    @abstractmethod
    def _get_score_logger(run: runAdapterT) -> ArtifactLogger[float, runAdapterT]: ...

    @staticmethod
    @abstractmethod
    def _get_array_logger(
        run: runAdapterT,
    ) -> ArtifactLogger[ndarray, runAdapterT]: ...

    @staticmethod
    @abstractmethod
    def _get_plot_logger(run: runAdapterT) -> ArtifactLogger[Figure, runAdapterT]: ...

    @staticmethod
    @abstractmethod
    def _get_score_collection_logger(
        run: runAdapterT,
    ) -> ArtifactLogger[Dict[str, float], runAdapterT]: ...

    @staticmethod
    @abstractmethod
    def _get_array_collection_logger(
        run: runAdapterT,
    ) -> ArtifactLogger[Dict[str, ndarray], runAdapterT]: ...

    @staticmethod
    @abstractmethod
    def _get_plot_collection_logger(
        run: runAdapterT,
    ) -> ArtifactLogger[Dict[str, Figure], runAdapterT]: ...

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
