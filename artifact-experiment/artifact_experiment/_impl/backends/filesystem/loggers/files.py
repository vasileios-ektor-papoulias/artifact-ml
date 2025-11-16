import os

from artifact_experiment._base.primitives.file import File
from artifact_experiment._impl.backends.filesystem.loggers.base import FilesystemLogger
from artifact_experiment._utils.filesystem.incremental_paths import IncrementalPathGenerator


class FilesystemFileLogger(FilesystemLogger[File]):
    _root_dir = "files"

    def _append(self, item_path: str, item: File):
        item_path = IncrementalPathGenerator.generate(dir_path=item_path)
        self._run.log_file(backend_dir=item_path, file=item)

    @classmethod
    def _get_relative_path(cls, item_name: str) -> str:
        return item_name

    def _get_root_dir(self) -> str:
        return os.path.join(self._run.run_dir, "files")
