from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from artifact_core.typing import (
    Array,
    ArrayCollection,
    ArtifactResult,
    Plot,
    PlotCollection,
    Score,
    ScoreCollection,
)

from artifact_experiment._base.primitives.file import File
from artifact_experiment._base.tracking.background.item import (
    ArrayCollectionQueueItem,
    ArrayQueueItem,
    FileQueueItem,
    PlotCollectionQueueItem,
    PlotQueueItem,
    ScoreCollectionQueueItem,
    ScoreQueueItem,
    TrackingQueueItem,
)
from artifact_experiment._base.tracking.background.temp_dir import TrackingTempDir
from artifact_experiment._base.typing.tracking_data import TrackingData
from artifact_experiment._utils.concurrency.typed_queue import TypedQueue

TrackingDataTContr = TypeVar("TrackingDataTContr", bound=TrackingData, contravariant=True)


class TrackingQueueWriter(ABC, Generic[TrackingDataTContr]):
    def __init__(self, queue: TypedQueue[TrackingQueueItem[TrackingData]]):
        self._queue = queue

    def write(self, name: str, value: TrackingDataTContr):
        queue_item = self._get_queue_item(name=name, value=value)
        self._queue.put(queue_item)

    @classmethod
    @abstractmethod
    def _get_queue_item(
        cls, name: str, value: TrackingDataTContr
    ) -> TrackingQueueItem[TrackingDataTContr]: ...


ArtifactResultTContr = TypeVar("ArtifactResultTContr", bound=ArtifactResult, contravariant=True)


class ArtifactQueueWriter(TrackingQueueWriter[ArtifactResultTContr]):
    pass


class ScoreWriter(ArtifactQueueWriter[Score]):
    @classmethod
    def _get_queue_item(cls, name: str, value: Score) -> ScoreQueueItem:
        return ScoreQueueItem(name=name, value=value)


class ArrayWriter(ArtifactQueueWriter[Array]):
    @classmethod
    def _get_queue_item(cls, name: str, value: Array) -> ArrayQueueItem:
        return ArrayQueueItem(name=name, value=value)


class PlotWriter(ArtifactQueueWriter[Plot]):
    @classmethod
    def _get_queue_item(cls, name: str, value: Plot) -> PlotQueueItem:
        return PlotQueueItem(name=name, value=value)


class ScoreCollectionWriter(ArtifactQueueWriter[ScoreCollection]):
    @classmethod
    def _get_queue_item(cls, name: str, value: ScoreCollection) -> ScoreCollectionQueueItem:
        return ScoreCollectionQueueItem(name=name, value=value)


class ArrayCollectionWriter(ArtifactQueueWriter[ArrayCollection]):
    @classmethod
    def _get_queue_item(cls, name: str, value: ArrayCollection) -> ArrayCollectionQueueItem:
        return ArrayCollectionQueueItem(name=name, value=value)


class PlotCollectionWriter(ArtifactQueueWriter[PlotCollection]):
    @classmethod
    def _get_queue_item(cls, name: str, value: PlotCollection) -> PlotCollectionQueueItem:
        return PlotCollectionQueueItem(name=name, value=value)


class FileWriter(TrackingQueueWriter[File]):
    def __init__(
        self, queue: TypedQueue[TrackingQueueItem[TrackingData]], temp_dir: TrackingTempDir
    ):
        super().__init__(queue=queue)
        self._temp_dir = temp_dir

    def write(self, name: str, value: File):
        managed_path = self._temp_dir.copy_file(value.path_source)
        managed_file = File(path_source=managed_path.as_posix())
        queue_item = self._get_queue_item(name=name, value=managed_file)
        self._queue.put(queue_item)

    @classmethod
    def _get_queue_item(cls, name: str, value: File) -> FileQueueItem:
        return FileQueueItem(name=name, value=value)
