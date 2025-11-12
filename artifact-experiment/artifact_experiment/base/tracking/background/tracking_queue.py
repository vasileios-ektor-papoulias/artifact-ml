from typing import Dict, Type, TypeVar

from artifact_experiment.base.entities.file import File
from artifact_experiment.base.entities.tracking_data import TrackingData
from artifact_experiment.base.tracking.background.item import (
    TrackingQueueItem,
)
from artifact_experiment.base.tracking.background.queue import ThreadQueue
from artifact_experiment.base.tracking.background.temp_dir import ManagedTempDir
from artifact_experiment.base.tracking.background.writer import (
    ArrayCollectionWriter,
    ArrayWriter,
    FileWriter,
    PlotCollectionWriter,
    PlotWriter,
    ScoreCollectionWriter,
    ScoreWriter,
)

TrackingQueueT = TypeVar("TrackingQueueT", bound="TrackingQueue")


class TrackingQueue:
    def __init__(
        self,
        queue: ThreadQueue[TrackingQueueItem[TrackingData]],
        score_writer: ScoreWriter,
        array_writer: ArrayWriter,
        plot_writer: PlotWriter,
        score_collection_writer: ScoreCollectionWriter,
        array_collection_writer: ArrayCollectionWriter,
        plot_collection_writer: PlotCollectionWriter,
        file_writer: FileWriter,
        managed_temp_dir: ManagedTempDir,
    ):
        self._queue = queue
        self._score_writer = score_writer
        self._array_writer = array_writer
        self._plot_writer = plot_writer
        self._score_collection_writer = score_collection_writer
        self._array_collection_writer = array_collection_writer
        self._plot_collection_writer = plot_collection_writer
        self._file_writer = file_writer
        self._managed_temp_dir = managed_temp_dir

    @classmethod
    def build(cls: Type[TrackingQueueT]) -> TrackingQueueT:
        queue: ThreadQueue[TrackingQueueItem[TrackingData]] = ThreadQueue()
        managed_temp_dir = ManagedTempDir()
        score_writer = ScoreWriter(queue=queue)
        array_writer = ArrayWriter(queue=queue)
        plot_writer = PlotWriter(queue=queue)
        score_collection_writer = ScoreCollectionWriter(queue=queue)
        array_collection_writer = ArrayCollectionWriter(queue=queue)
        plot_collection_writer = PlotCollectionWriter(queue=queue)
        file_writer = FileWriter(queue=queue, managed_temp_dir=managed_temp_dir)
        return cls(
            queue=queue,
            score_writer=score_writer,
            array_writer=array_writer,
            plot_writer=plot_writer,
            score_collection_writer=score_collection_writer,
            array_collection_writer=array_collection_writer,
            plot_collection_writer=plot_collection_writer,
            file_writer=file_writer,
            managed_temp_dir=managed_temp_dir,
        )

    @property
    def queue(self) -> ThreadQueue[TrackingQueueItem[TrackingData]]:
        return self._queue

    @property
    def score_writer(self) -> ScoreWriter:
        return self._score_writer

    @property
    def array_writer(self) -> ArrayWriter:
        return self._array_writer

    @property
    def plot_writer(self) -> PlotWriter:
        return self._plot_writer

    @property
    def score_collection_writer(self) -> ScoreCollectionWriter:
        return self._score_collection_writer

    @property
    def array_collection_writer(self) -> ArrayCollectionWriter:
        return self._array_collection_writer

    @property
    def plot_collection_writer(self) -> PlotCollectionWriter:
        return self._plot_collection_writer

    @property
    def file_writer(self) -> FileWriter:
        return self._file_writer

    @property
    def managed_temp_dir(self) -> ManagedTempDir:
        return self._managed_temp_dir

    def put_score(self, name: str, value: float):
        self._score_writer.write(name=name, value=value)

    def put_array(self, name: str, value: Array):
        self._array_writer.write(name=name, value=value)

    def put_plot(self, name: str, value: Figure):
        self._plot_writer.write(name=name, value=value)

    def put_score_collection(self, name: str, value: Dict[str, float]):
        self._score_collection_writer.write(name=name, value=value)

    def put_array_collection(self, name: str, value: Dict[str, Array]):
        self._array_collection_writer.write(name=name, value=value)

    def put_plot_collection(self, name: str, value: Dict[str, Figure]):
        self._plot_collection_writer.write(name=name, value=value)

    def put_file(self, name: str, value: File):
        self._file_writer.write(name=name, value=value)
