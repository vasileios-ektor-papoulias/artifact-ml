import os
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from artifact_core.typing import ArtifactResult

from artifact_experiment._base.tracking.backend.adapter import RunAdapter
from artifact_experiment._base.typing.tracking_data import TrackingData

RunAdapterT = TypeVar("RunAdapterT", bound=RunAdapter)
TrackingDataTContr = TypeVar("TrackingDataTContr", bound=TrackingData, contravariant=True)


class BackendLogger(ABC, Generic[RunAdapterT, TrackingDataTContr]):
    def __init__(self, run: RunAdapterT):
        self._run = run

    @abstractmethod
    def _append(self, item_path: str, item: TrackingDataTContr):
        pass

    @classmethod
    @abstractmethod
    def _get_relative_path(cls, item_name: str) -> str:
        pass

    @abstractmethod
    def _get_root_dir(self) -> str: ...

    @classmethod
    def _get_item_path(cls, root_dir: str, item_name: str) -> str:
        relative_path = cls._get_relative_path(item_name=item_name)
        item_path = os.path.join(root_dir, relative_path)
        return item_path

    def log(self, item_name: str, item: TrackingDataTContr):
        root_dir = self._get_root_dir()
        path = self._get_item_path(root_dir=root_dir, item_name=item_name)
        self._append(item_path=path, item=item)


ArtifactResultTContr = TypeVar("ArtifactResultTContr", bound=ArtifactResult, contravariant=True)


ArtifactLogger = BackendLogger[RunAdapterT, ArtifactResultTContr]
