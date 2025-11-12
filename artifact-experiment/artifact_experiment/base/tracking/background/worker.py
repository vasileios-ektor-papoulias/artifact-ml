from abc import ABC, abstractmethod
from threading import Thread
from typing import Dict, Optional

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
    StopFlag,
    TrackingQueueItem,
)
from artifact_experiment.base.tracking.background.queue import ThreadQueue
from artifact_experiment.base.tracking.background.temp_dir import ManagedTempDir


class TrackingWorker(ABC):
    _join_timeout_secs = 5

    def __init__(
        self,
        queue: ThreadQueue[TrackingQueueItem[TrackingData]],
        managed_temp_dir: ManagedTempDir,
    ):
        self._queue = queue
        self._managed_temp_dir = managed_temp_dir
        self._thread: Optional[Thread] = None
        self._stop_flag = False

    @abstractmethod
    def _log_score(self, name: str, value: float): ...

    @abstractmethod
    def _log_array(self, name: str, value: Array): ...

    @abstractmethod
    def _log_plot(self, name: str, value: Figure): ...

    @abstractmethod
    def _log_score_collection(self, name: str, value: Dict[str, float]): ...

    @abstractmethod
    def _log_array_collection(self, name: str, value: Dict[str, Array]): ...

    @abstractmethod
    def _log_plot_collection(self, name: str, value: Dict[str, Figure]): ...

    @abstractmethod
    def _log_file(self, name: str, value: File): ...

    def start(self):
        if self._thread is None or not self._thread.is_alive():
            self._stop_flag = False
            self._thread = Thread(target=self._start, daemon=True)
            self._thread.start()

    def stop(self):
        if self._thread is not None and self._thread.is_alive():
            self._stop_flag = True
            try:
                self._queue.put(StopFlag(), block=False)
            except Exception:
                pass
            self._thread.join(timeout=self._join_timeout_secs)
            self._managed_temp_dir.cleanup()
            self._thread = None

    def _start(self):
        while not self._stop_flag:
            queue_item: TrackingQueueItem[TrackingData] = self._queue.get()
            if self._should_stop(queue_item=queue_item):
                break
            self._process_item(queue_item=queue_item)

    def _process_item(self, queue_item: TrackingQueueItem[TrackingData]):
        if isinstance(queue_item, ScoreQueueItem):
            self._log_score(name=queue_item.name, value=queue_item.value)
        elif isinstance(queue_item, ArrayQueueItem):
            self._log_array(name=queue_item.name, value=queue_item.value)
        elif isinstance(queue_item, PlotQueueItem):
            self._log_plot(name=queue_item.name, value=queue_item.value)
        elif isinstance(queue_item, ScoreCollectionQueueItem):
            self._log_score_collection(name=queue_item.name, value=queue_item.value)
        elif isinstance(queue_item, ArrayCollectionQueueItem):
            self._log_array_collection(name=queue_item.name, value=queue_item.value)
        elif isinstance(queue_item, PlotCollectionQueueItem):
            self._log_plot_collection(name=queue_item.name, value=queue_item.value)
        elif isinstance(queue_item, FileQueueItem):
            self._log_file(name=queue_item.name, value=queue_item.value)
            self._managed_temp_dir.remove_file(queue_item.value.path_source)
        else:
            raise TypeError(f"Unsupported queue item type: {type(queue_item).__name__}. ")

    def _should_stop(self, queue_item: TrackingQueueItem[TrackingData]) -> bool:
        return isinstance(queue_item, StopFlag) or self._stop_flag
