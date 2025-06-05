from abc import abstractmethod
from typing import Generic, TypeVar

from artifact_experiment.base.tracking.client import TrackingClient

ExportDataT = TypeVar("ExportDataT")


class Exporter(Generic[ExportDataT]):
    @classmethod
    def export(cls, data: ExportDataT, tracking_client: TrackingClient, step: int, prefix: str):
        target_dir = cls._get_target_dir()
        filename = cls._get_filename(prefix=prefix, step=step)
        cls._export(
            data=data, tracking_client=tracking_client, target_dir=target_dir, filename=filename
        )

    @classmethod
    @abstractmethod
    def _export(
        cls, data: ExportDataT, tracking_client: TrackingClient, target_dir: str, filename: str
    ): ...

    @classmethod
    @abstractmethod
    def _get_target_dir(cls) -> str: ...

    @classmethod
    def _get_filename(cls, prefix: str, step: int) -> str:
        if prefix:
            filename = f"{prefix}_{step}.csv"
        else:
            filename = f"{step}.csv"
        return filename
