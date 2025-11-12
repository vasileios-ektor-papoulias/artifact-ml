from abc import abstractmethod
from typing import Dict, Generic, Type, TypeVar

from artifact_experiment.base.entities.file import File
from artifact_experiment.base.entities.tracking_data import TrackingData
from artifact_experiment.base.tracking.backend.adapter import RunAdapter
from artifact_experiment.base.tracking.backend.logger import ArtifactLogger, BackendLogger
from artifact_experiment.base.tracking.background.item import TrackingQueueItem
from artifact_experiment.base.tracking.background.queue import ThreadQueue
from artifact_experiment.base.tracking.background.temp_dir import ManagedTempDir
from artifact_experiment.base.tracking.background.worker import TrackingWorker

RunAdapterT = TypeVar("RunAdapterT", bound=RunAdapter)
LoggerWorkerT = TypeVar("LoggerWorkerT", bound="LoggerWorker")


class LoggerWorker(TrackingWorker, Generic[RunAdapterT]):
    def __init__(
        self,
        run: RunAdapterT,
        score_logger: ArtifactLogger[float, RunAdapterT],
        array_logger: ArtifactLogger[Array, RunAdapterT],
        plot_logger: ArtifactLogger[Figure, RunAdapterT],
        score_collection_logger: ArtifactLogger[Dict[str, float], RunAdapterT],
        array_collection_logger: ArtifactLogger[Dict[str, Array], RunAdapterT],
        plot_collection_logger: ArtifactLogger[Dict[str, Figure], RunAdapterT],
        file_logger: BackendLogger[File, RunAdapterT],
        queue: ThreadQueue[TrackingQueueItem[TrackingData]],
        managed_temp_dir: ManagedTempDir,
    ):
        super().__init__(queue=queue, managed_temp_dir=managed_temp_dir)
        self._run = run
        self._score_logger = score_logger
        self._array_logger = array_logger
        self._plot_logger = plot_logger
        self._score_collection_logger = score_collection_logger
        self._array_collection_logger = array_collection_logger
        self._plot_collection_logger = plot_collection_logger
        self._file_logger = file_logger

    @classmethod
    def build(
        cls: Type[LoggerWorkerT],
        run: RunAdapterT,
        queue: ThreadQueue[TrackingQueueItem[TrackingData]],
        managed_temp_dir: ManagedTempDir,
    ) -> LoggerWorkerT:
        score_logger = cls._get_score_logger(run=run)
        array_logger = cls._get_array_logger(run=run)
        plot_logger = cls._get_plot_logger(run=run)
        score_collection_logger = cls._get_score_collection_logger(run=run)
        array_collection_logger = cls._get_array_collection_logger(run=run)
        plot_collection_logger = cls._get_plot_collection_logger(run=run)
        file_logger = cls._get_file_logger(run=run)
        return cls(
            run=run,
            score_logger=score_logger,
            array_logger=array_logger,
            plot_logger=plot_logger,
            score_collection_logger=score_collection_logger,
            array_collection_logger=array_collection_logger,
            plot_collection_logger=plot_collection_logger,
            file_logger=file_logger,
            queue=queue,
            managed_temp_dir=managed_temp_dir,
        )

    @staticmethod
    @abstractmethod
    def _get_score_logger(
        run: RunAdapterT,
    ) -> ArtifactLogger[float, RunAdapterT]: ...

    @staticmethod
    @abstractmethod
    def _get_array_logger(
        run: RunAdapterT,
    ) -> ArtifactLogger[Array, RunAdapterT]: ...

    @staticmethod
    @abstractmethod
    def _get_plot_logger(
        run: RunAdapterT,
    ) -> ArtifactLogger[Figure, RunAdapterT]: ...

    @staticmethod
    @abstractmethod
    def _get_score_collection_logger(
        run: RunAdapterT,
    ) -> ArtifactLogger[Dict[str, float], RunAdapterT]: ...

    @staticmethod
    @abstractmethod
    def _get_array_collection_logger(
        run: RunAdapterT,
    ) -> ArtifactLogger[Dict[str, Array], RunAdapterT]: ...

    @staticmethod
    @abstractmethod
    def _get_plot_collection_logger(
        run: RunAdapterT,
    ) -> ArtifactLogger[Dict[str, Figure], RunAdapterT]: ...

    @staticmethod
    @abstractmethod
    def _get_file_logger(
        run: RunAdapterT,
    ) -> BackendLogger[File, RunAdapterT]: ...

    def _log_score(self, name: str, value: float):
        self._score_logger.log(item_name=name, item=value)

    def _log_array(self, name: str, value: Array):
        self._array_logger.log(item_name=name, item=value)

    def _log_plot(self, name: str, value: Figure):
        self._plot_logger.log(item_name=name, item=value)

    def _log_score_collection(self, name: str, value: Dict[str, float]):
        self._score_collection_logger.log(item_name=name, item=value)

    def _log_array_collection(self, name: str, value: Dict[str, Array]):
        self._array_collection_logger.log(item_name=name, item=value)

    def _log_plot_collection(self, name: str, value: Dict[str, Figure]):
        self._plot_collection_logger.log(item_name=name, item=value)

    def _log_file(self, name: str, value: File):
        self._file_logger.log(item_name=name, item=value)
