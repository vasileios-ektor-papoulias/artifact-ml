from queue import Queue
from typing import Generic, Optional, TypeVar

QueueDataT = TypeVar("QueueDataT")


class TypedQueue(Generic[QueueDataT]):
    def __init__(self):
        self._queue: Queue[QueueDataT] = Queue()

    def put(self, item: QueueDataT, block: bool = True, timeout: Optional[float] = None) -> None:
        self._queue.put(item, block=block, timeout=timeout)

    def get(self, block: bool = True, timeout: Optional[float] = None) -> QueueDataT:
        return self._queue.get(block=block, timeout=timeout)

    def empty(self) -> bool:
        return self._queue.empty()

    def qsize(self) -> int:
        return self._queue.qsize()

    def task_done(self) -> None:
        self._queue.task_done()

    def join(self) -> None:
        self._queue.join()
