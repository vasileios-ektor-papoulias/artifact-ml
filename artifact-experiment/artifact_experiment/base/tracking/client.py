from abc import abstractmethod
from typing import Dict, Generic, Type, TypeVar

from matplotlib.figure import Figure
from numpy import ndarray

from artifact_experiment.base.tracking.adapter import RunAdapter
from artifact_experiment.base.tracking.logger import ArtifactLogger

RunAdapterT = TypeVar("RunAdapterT", bound=RunAdapter)
TrackingClientT = TypeVar("TrackingClientT", bound="TrackingClient")


class TrackingClient(Generic[RunAdapterT]):
    def __init__(
        self,
        run: RunAdapterT,
        score_logger: ArtifactLogger[float, RunAdapterT],
        array_logger: ArtifactLogger[ndarray, RunAdapterT],
        plot_logger: ArtifactLogger[Figure, RunAdapterT],
        score_collection_logger: ArtifactLogger[Dict[str, float], RunAdapterT],
        array_collection_logger: ArtifactLogger[Dict[str, ndarray], RunAdapterT],
        plot_collection_logger: ArtifactLogger[Dict[str, Figure], RunAdapterT],
    ):
        self._run = run
        self._score_logger = score_logger
        self._array_logger = array_logger
        self._plot_logger = plot_logger
        self._score_collection_logger = score_collection_logger
        self._array_collection_logger = array_collection_logger
        self._plot_collection_logger = plot_collection_logger

    @classmethod
    def from_run(cls: Type[TrackingClientT], run: RunAdapterT) -> TrackingClientT:
        client = cls._build(run=run)
        return client

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

    @classmethod
    def _build(cls: Type[TrackingClientT], run: RunAdapterT) -> TrackingClientT:
        score_logger = cls._get_score_logger(run=run)
        array_logger = cls._get_array_logger(run=run)
        plot_logger = cls._get_plot_logger(run=run)
        score_collection_logger = cls._get_score_collection_logger(run=run)
        array_collection_logger = cls._get_array_collection_logger(run=run)
        plot_collection_logger = cls._get_plot_collection_logger(run=run)
        client = cls(
            run=run,
            score_logger=score_logger,
            array_logger=array_logger,
            plot_logger=plot_logger,
            score_collection_logger=score_collection_logger,
            array_collection_logger=array_collection_logger,
            plot_collection_logger=plot_collection_logger,
        )
        return client
