import os
import tempfile
from abc import abstractmethod
from typing import Any, Generic, Optional, TypeVar

from artifact_experiment._base.components.callbacks.tracking import TrackingCallback
from artifact_experiment._base.components.resources.export import ExportCallbackResources
from artifact_experiment._base.primitives.file import File
from artifact_experiment._base.tracking.background.writer import FileWriter
from artifact_experiment._utils.filesystem.extension_normalizer import ExtensionNormalizer

ExportCallbackResourcesT = TypeVar(
    "ExportCallbackResourcesT", bound=ExportCallbackResources[Any], contravariant=True
)


class ExportCallback(
    TrackingCallback[ExportCallbackResourcesT, File],
    Generic[ExportCallbackResourcesT],
):
    def __init__(self, writer: FileWriter):
        super().__init__(base_key=self._get_export_name())
        self._writer = writer
        self._extension = self._get_extension()
        self._tmpdir: Optional[str] = None

    @classmethod
    @abstractmethod
    def _get_export_name(cls) -> str: ...

    @classmethod
    @abstractmethod
    def _get_extension(cls) -> str: ...

    @classmethod
    @abstractmethod
    def _save_local(cls, resources: ExportCallbackResourcesT, filepath: str) -> str: ...

    def execute(self, resources: ExportCallbackResourcesT):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._tmpdir = tmpdir
            super().execute(resources=resources)

    def _compute(self, resources: ExportCallbackResourcesT) -> File:
        assert self._tmpdir is not None
        filepath = self._get_filepath(
            export_dir=self._tmpdir, filename=self.key, extension=self._extension
        )
        filepath = self._save_local(resources=resources, filepath=filepath)
        file = File(path_source=filepath)
        return file

    @staticmethod
    def _get_filepath(export_dir: str, filename: str, extension: str) -> str:
        extension = ExtensionNormalizer.normalize(extension=extension)
        return os.path.join(export_dir, f"{filename}{extension}")
