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
from artifact_experiment._base.tracking.backend.worker import BackendLoggingWorker
from artifact_experiment._base.tracking.background.queue import TrackingQueue
from artifact_experiment._base.tracking.background.writer import (
    ArrayCollectionWriter,
    ArrayWriter,
    FileWriter,
    PlotCollectionWriter,
    PlotWriter,
    ScoreCollectionWriter,
    ScoreWriter,
)

RunAdapterT = TypeVar("RunAdapterT", bound=RunAdapter)
TrackingClientT = TypeVar("TrackingClientT", bound="TrackingClient")


class TrackingClient(Generic[RunAdapterT]):
    def __init__(
        self,
        run: RunAdapterT,
        tracking_queue: TrackingQueue,
        logger_worker: BackendLoggingWorker[RunAdapterT],
    ):
        self._run = run
        self._tracking_queue = tracking_queue
        self._logger_worker = logger_worker
        self._logger_worker.start()

    @classmethod
    def from_run(cls: Type[TrackingClientT], run: RunAdapterT) -> TrackingClientT:
        client = cls._build(run=run)
        return client

    @property
    def run(self) -> RunAdapterT:
        return self._run

    @property
    def queue(self) -> TrackingQueue:
        return self._tracking_queue

    @property
    def score_writer(self) -> ScoreWriter:
        return self._tracking_queue.score_writer

    @property
    def array_writer(self) -> ArrayWriter:
        return self._tracking_queue.array_writer

    @property
    def plot_writer(self) -> PlotWriter:
        return self._tracking_queue.plot_writer

    @property
    def score_collection_writer(self) -> ScoreCollectionWriter:
        return self._tracking_queue.score_collection_writer

    @property
    def array_collection_writer(self) -> ArrayCollectionWriter:
        return self._tracking_queue.array_collection_writer

    @property
    def plot_collection_writer(self) -> PlotCollectionWriter:
        return self._tracking_queue.plot_collection_writer

    @property
    def file_writer(self) -> FileWriter:
        return self._tracking_queue.file_writer

    @staticmethod
    @abstractmethod
    def _get_worker(
        run: RunAdapterT, tracking_queue: TrackingQueue
    ) -> BackendLoggingWorker[RunAdapterT]: ...

    def log_score(self, score: Score, name: str):
        self._tracking_queue.put_score(name=name, value=score)

    def log_array(self, array: Array, name: str):
        self._tracking_queue.put_array(name=name, value=array)

    def log_plot(self, plot: Plot, name: str):
        self._tracking_queue.put_plot(name=name, value=plot)

    def log_score_collection(self, score_collection: ScoreCollection, name: str):
        self._tracking_queue.put_score_collection(name=name, value=score_collection)

    def log_array_collection(self, array_collection: ArrayCollection, name: str):
        self._tracking_queue.put_array_collection(name=name, value=array_collection)

    def log_plot_collection(self, plot_collection: PlotCollection, name: str):
        self._tracking_queue.put_plot_collection(name=name, value=plot_collection)

    def log_file(self, path_source: str, name: str):
        file = File(path_source=path_source)
        self._tracking_queue.put_file(name=name, value=file)

    def stop(self):
        self._logger_worker.stop()
        self._run.stop()

    @classmethod
    def _build(cls: Type[TrackingClientT], run: RunAdapterT) -> TrackingClientT:
        tracking_queue = TrackingQueue.build()
        logger_worker = cls._get_worker(tracking_queue=tracking_queue, run=run)
        client = cls(
            run=run,
            tracking_queue=tracking_queue,
            logger_worker=logger_worker,
        )
        return client
