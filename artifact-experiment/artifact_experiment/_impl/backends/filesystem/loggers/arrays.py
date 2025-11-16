import os

import numpy as np
from artifact_core.typing import Array

from artifact_experiment._impl.backends.filesystem.adapter import InactiveFilesystemRunError
from artifact_experiment._impl.backends.filesystem.loggers.artifacts import FilesystemArtifactLogger
from artifact_experiment._utils.filesystem.incremental_paths import IncrementalPathGenerator


class FilesystemArrayLogger(FilesystemArtifactLogger[Array]):
    _ext: str = "npy"

    def _append(self, item_path: str, item: Array):
        if self._run.is_active:
            self._export_array(dir_path=item_path, array=item)
        else:
            raise InactiveFilesystemRunError("Run is inactive")

    @classmethod
    def _export_array(cls, dir_path: str, array: Array):
        filepath = IncrementalPathGenerator.generate(dir_path=dir_path, ext=cls._ext)
        np.save(file=filepath, arr=array)

    @classmethod
    def _get_relative_path(cls, item_name: str) -> str:
        return os.path.join("arrays", item_name)
