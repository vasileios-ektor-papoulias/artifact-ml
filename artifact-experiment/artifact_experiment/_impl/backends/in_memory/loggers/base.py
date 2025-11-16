from abc import abstractmethod
from typing import Generic, TypeVar

from artifact_experiment._base.tracking.backend.logger import BackendLogger
from artifact_experiment._base.typing.tracking_data import TrackingData
from artifact_experiment._impl.backends.in_memory.adapter import InMemoryRunAdapter

TrackingDataT = TypeVar("TrackingDataT", bound=TrackingData)


class InMemoryLogger(BackendLogger[InMemoryRunAdapter, TrackingDataT], Generic[TrackingDataT]):
    @abstractmethod
    def _append(self, item_path: str, item: TrackingDataT): ...

    @classmethod
    @abstractmethod
    def _get_relative_path(cls, item_name: str) -> str: ...

    def _get_root_dir(self) -> str:
        return f"{self._run.experiment_id}/{self._run.run_id}"

    @staticmethod
    def _get_store_key(item_path: str, step: int) -> str:
        return f"{item_path}/{step}"
