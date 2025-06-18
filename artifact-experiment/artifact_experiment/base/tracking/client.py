from abc import abstractmethod
from typing import Dict, Generic, TypeVar

from matplotlib.figure import Figure
from numpy import ndarray

from artifact_experiment.base.tracking.adapter import RunAdapter
from artifact_experiment.base.tracking.logger import ArtifactLogger

RunAdapterT = TypeVar("RunAdapterT", bound=RunAdapter)
TrackingClientT = TypeVar("TrackingClientT", bound="TrackingClient")


class TrackingClient(Generic[RunAdapterT]):
    def __init__(self, run: RunAdapterT):
        self._run = run
        self._score_logger = self._get_score_logger(run=self._run)
        self._array_logger = self._get_array_logger(run=self._run)
        self._plot_logger = self._get_plot_logger(run=self._run)
        self._score_collection_logger = self._get_score_collection_logger(run=self._run)
        self._array_collection_logger = self._get_array_collection_logger(run=self._run)
        self._plot_collection_logger = self._get_plot_collection_logger(run=self._run)

    @property
    def run(self) -> RunAdapterT:
        return self._run

    @run.setter
    def run(self, run: RunAdapterT):
        self._run = run

    @staticmethod
    @abstractmethod
    def _get_score_logger(run: RunAdapterT) -> ArtifactLogger[float, RunAdapterT]: ...

    @staticmethod
    @abstractmethod
    def _get_array_logger(
        run: RunAdapterT,
    ) -> ArtifactLogger[ndarray, RunAdapterT]: ...

    @staticmethod
    @abstractmethod
    def _get_plot_logger(run: RunAdapterT) -> ArtifactLogger[Figure, RunAdapterT]: ...

    @staticmethod
    @abstractmethod
    def _get_score_collection_logger(
        run: RunAdapterT,
    ) -> ArtifactLogger[Dict[str, float], RunAdapterT]: ...

    @staticmethod
    @abstractmethod
    def _get_array_collection_logger(
        run: RunAdapterT,
    ) -> ArtifactLogger[Dict[str, ndarray], RunAdapterT]: ...

    @staticmethod
    @abstractmethod
    def _get_plot_collection_logger(
        run: RunAdapterT,
    ) -> ArtifactLogger[Dict[str, Figure], RunAdapterT]: ...

    def log_score(self, score: float, name: str):
        self._score_logger.log(artifact=score, artifact_name=name)

    def log_array(self, array: ndarray, name: str):
        self._array_logger.log(artifact_name=name, artifact=array)

    def log_plot(self, plot: Figure, name: str):
        self._plot_logger.log(artifact_name=name, artifact=plot)

    def log_score_collection(self, score_collection: Dict[str, float], name: str):
        self._score_collection_logger.log(artifact_name=name, artifact=score_collection)

    def log_array_collection(self, array_collection: Dict[str, ndarray], name: str):
        self._array_collection_logger.log(artifact_name=name, artifact=array_collection)

    def log_plot_collection(self, plot_collection: Dict[str, Figure], name: str):
        self._plot_collection_logger.log(artifact_name=name, artifact=plot_collection)

    def upload(self, path_source: str, dir_target: str):
        self._run.upload(path_source=path_source, dir_target=dir_target)
