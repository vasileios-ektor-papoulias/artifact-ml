from abc import abstractmethod
from typing import Generic, Type, TypeVar

from artifact_core.typing import (
    Array,
    ArrayCollection,
    Plot,
    PlotCollection,
    Score,
    ScoreCollection,
)

from artifact_experiment._base.primitives.file import File
from artifact_experiment._base.tracking.backend.adapter import RunAdapter
from artifact_experiment._base.tracking.backend.logger import ArtifactLogger, BackendLogger
from artifact_experiment._base.tracking.background.item import TrackingQueueItem
from artifact_experiment._base.tracking.background.temp_dir import TrackingTempDir
from artifact_experiment._base.tracking.background.worker import TrackingWorker
from artifact_experiment._base.typing.tracking_data import TrackingData
from artifact_experiment._utils.concurrency.typed_queue import TypedQueue

RunAdapterT = TypeVar("RunAdapterT", bound=RunAdapter)
BackendLoggingWorkerT = TypeVar("BackendLoggingWorkerT", bound="BackendLoggingWorker")


class BackendLoggingWorker(TrackingWorker, Generic[RunAdapterT]):
    def __init__(
        self,
        run: RunAdapterT,
        score_logger: ArtifactLogger[RunAdapterT, Score],
        array_logger: ArtifactLogger[RunAdapterT, Array],
        plot_logger: ArtifactLogger[RunAdapterT, Plot],
        score_collection_logger: ArtifactLogger[RunAdapterT, ScoreCollection],
        array_collection_logger: ArtifactLogger[RunAdapterT, ArrayCollection],
        plot_collection_logger: ArtifactLogger[RunAdapterT, PlotCollection],
        file_logger: BackendLogger[RunAdapterT, File],
        queue: TypedQueue[TrackingQueueItem[TrackingData]],
        temp_dir: TrackingTempDir,
    ):
        super().__init__(queue=queue, temp_dir=temp_dir)
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
        cls: Type[BackendLoggingWorkerT],
        run: RunAdapterT,
        queue: TypedQueue[TrackingQueueItem[TrackingData]],
        temp_dir: TrackingTempDir,
    ) -> BackendLoggingWorkerT:
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
            temp_dir=temp_dir,
        )

    @staticmethod
    @abstractmethod
    def _get_score_logger(
        run: RunAdapterT,
    ) -> ArtifactLogger[RunAdapterT, Score]: ...

    @staticmethod
    @abstractmethod
    def _get_array_logger(
        run: RunAdapterT,
    ) -> ArtifactLogger[RunAdapterT, Array]: ...

    @staticmethod
    @abstractmethod
    def _get_plot_logger(
        run: RunAdapterT,
    ) -> ArtifactLogger[RunAdapterT, Plot]: ...

    @staticmethod
    @abstractmethod
    def _get_score_collection_logger(
        run: RunAdapterT,
    ) -> ArtifactLogger[RunAdapterT, ScoreCollection]: ...

    @staticmethod
    @abstractmethod
    def _get_array_collection_logger(
        run: RunAdapterT,
    ) -> ArtifactLogger[RunAdapterT, ArrayCollection]: ...

    @staticmethod
    @abstractmethod
    def _get_plot_collection_logger(
        run: RunAdapterT,
    ) -> ArtifactLogger[RunAdapterT, PlotCollection]: ...

    @staticmethod
    @abstractmethod
    def _get_file_logger(
        run: RunAdapterT,
    ) -> BackendLogger[RunAdapterT, File]: ...

    def _log_score(self, name: str, value: float):
        self._score_logger.log(item_name=name, item=value)

    def _log_array(self, name: str, value: Array):
        self._array_logger.log(item_name=name, item=value)

    def _log_plot(self, name: str, value: Plot):
        self._plot_logger.log(item_name=name, item=value)

    def _log_score_collection(self, name: str, value: ScoreCollection):
        self._score_collection_logger.log(item_name=name, item=value)

    def _log_array_collection(self, name: str, value: ArrayCollection):
        self._array_collection_logger.log(item_name=name, item=value)

    def _log_plot_collection(self, name: str, value: PlotCollection):
        self._plot_collection_logger.log(item_name=name, item=value)

    def _log_file(self, name: str, value: File):
        self._file_logger.log(item_name=name, item=value)
