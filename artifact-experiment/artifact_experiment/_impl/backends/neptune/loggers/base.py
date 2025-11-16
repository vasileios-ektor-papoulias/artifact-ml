from abc import abstractmethod
from typing import Generic, TypeVar

from artifact_experiment._base.tracking.backend.logger import BackendLogger
from artifact_experiment._base.typing.tracking_data import TrackingData
from artifact_experiment._impl.backends.neptune.adapter import NeptuneRunAdapter

TrackingDataT = TypeVar("TrackingDataT", bound=TrackingData)


class NeptuneLogger(BackendLogger[NeptuneRunAdapter, TrackingDataT], Generic[TrackingDataT]):
    @abstractmethod
    def _append(self, item_path: str, item: TrackingDataT): ...

    @classmethod
    @abstractmethod
    def _get_relative_path(cls, item_name: str) -> str: ...

    @abstractmethod
    def _get_root_dir(self) -> str: ...
