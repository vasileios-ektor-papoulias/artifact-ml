from abc import ABC, abstractmethod
from typing import Dict, Generic, TypeVar

from artifact_core._base.primitives import ArtifactResult

from artifact_experiment.base.entities.file import File
from artifact_experiment.base.entities.tracking_data import TrackingData
from artifact_experiment.base.tracking.background.item import (
    ArrayCollectionQueueItem,
    ArrayQueueItem,
    FileQueueItem,
    PlotCollectionQueueItem,
    PlotQueueItem,
    ScoreCollectionQueueItem,
    ScoreQueueItem,
    TrackingQueueItem,
)
from artifact_experiment.base.tracking.background.queue import ThreadQueue
from artifact_experiment.base.tracking.background.temp_dir import ManagedTempDir

TrackingDataTContr = TypeVar("TrackingDataTContr", bound=TrackingData, contravariant=True)


class TrackingQueueWriter(ABC, Generic[TrackingDataTContr]):
    def __init__(self, queue: ThreadQueue[TrackingQueueItem[TrackingData]]):
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


class ScoreWriter(ArtifactQueueWriter[float]):
    @classmethod
    def _get_queue_item(cls, name: str, value: float) -> ScoreQueueItem:
        return ScoreQueueItem(name=name, value=value)


class ArrayWriter(ArtifactQueueWriter[Array]):
    @classmethod
    def _get_queue_item(cls, name: str, value: Array) -> ArrayQueueItem:
        return ArrayQueueItem(name=name, value=value)


class PlotWriter(ArtifactQueueWriter[Figure]):
    @classmethod
    def _get_queue_item(cls, name: str, value: Figure) -> PlotQueueItem:
        return PlotQueueItem(name=name, value=value)


class ScoreCollectionWriter(ArtifactQueueWriter[Dict[str, float]]):
    @classmethod
    def _get_queue_item(cls, name: str, value: Dict[str, float]) -> ScoreCollectionQueueItem:
        return ScoreCollectionQueueItem(name=name, value=value)


class ArrayCollectionWriter(ArtifactQueueWriter[Dict[str, Array]]):
    @classmethod
    def _get_queue_item(cls, name: str, value: Dict[str, Array]) -> ArrayCollectionQueueItem:
        return ArrayCollectionQueueItem(name=name, value=value)


class PlotCollectionWriter(ArtifactQueueWriter[Dict[str, Figure]]):
    @classmethod
    def _get_queue_item(cls, name: str, value: Dict[str, Figure]) -> PlotCollectionQueueItem:
        return PlotCollectionQueueItem(name=name, value=value)


class FileWriter(TrackingQueueWriter[File]):
    def __init__(
        self,
        queue: ThreadQueue[TrackingQueueItem[TrackingData]],
        managed_temp_dir: ManagedTempDir,
    ):
        super().__init__(queue=queue)
        self._managed_temp_dir = managed_temp_dir

    def write(self, name: str, value: File):
        managed_path = self._managed_temp_dir.copy_file(value.path_source)
        managed_file = File(path_source=managed_path)
        queue_item = self._get_queue_item(name=name, value=managed_file)
        self._queue.put(queue_item)

    @classmethod
    def _get_queue_item(cls, name: str, value: File) -> FileQueueItem:
        return FileQueueItem(name=name, value=value)
