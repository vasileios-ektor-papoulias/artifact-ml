import tempfile
from abc import abstractmethod
from dataclasses import dataclass
from typing import Generic, Optional, TypeVar

from artifact_experiment.base.entities.file import File
from artifact_experiment.base.tracking.background.writer import FileWriter

from artifact_torch.base.components.callbacks.periodic import (
    PeriodicTrackingCallback,
    PeriodicTrackingCallbackResources,
)

ExportDataTCov = TypeVar("ExportDataTCov", covariant=True)


@dataclass(frozen=True)
class ExportCallbackResources(PeriodicTrackingCallbackResources, Generic[ExportDataTCov]):
    export_data: ExportDataTCov


ExportDataTContr = TypeVar("ExportDataTContr", contravariant=True)


class ExportCallback(
    PeriodicTrackingCallback[ExportCallbackResources[ExportDataTContr], File],
    Generic[ExportDataTContr],
):
    def __init__(self, period: int, writer: FileWriter):
        base_key = self._get_export_name()
        super().__init__(base_key=base_key, period=period)
        self._writer = writer
        self._tmpdir: Optional[str] = None

    @classmethod
    @abstractmethod
    def _get_export_name(cls) -> str: ...

    @classmethod
    @abstractmethod
    def _get_extension(cls) -> str: ...

    @classmethod
    @abstractmethod
    def _save_local(cls, data: ExportDataTContr, dir_target: str, filename: str) -> str: ...

    def execute(self, resources):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._tmpdir = tmpdir
            super().execute(resources=resources)

    def _compute(self, resources: ExportCallbackResources[ExportDataTContr]) -> File:
        data = resources.export_data
        filename = self._get_filename(base_name=self.key, step=resources.step)
        assert self._tmpdir is not None
        filepath = self._save_local(data=data, dir_target=self._tmpdir, filename=filename)
        file = File(path_source=filepath)
        return file

    @classmethod
    def _get_filename(cls, base_name: str, step: int) -> str:
        extension = cls._get_extension()
        extension = cls._normalize_extension(extension=extension)
        if base_name:
            filename = f"{base_name}_{step}{extension}"
        else:
            filename = f"{step}{extension}"
        return filename

    @staticmethod
    def _normalize_extension(extension: str) -> str:
        if not extension.startswith("."):
            return f".{extension}"
        return extension
