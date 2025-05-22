import os

import numpy as np
from numpy import ndarray

from artifact_experiment.libs.tracking.filesystem.adapter import (
    InactiveFilesystemRunError,
)
from artifact_experiment.libs.tracking.filesystem.loggers.base import FilesystemArtifactLogger
from artifact_experiment.libs.utils.filesystem import (
    IncrementalPathGenerator,
)


class FilesystemArrayLogger(FilesystemArtifactLogger[ndarray]):
    _fmt: str = "npy"

    def _log(self, path: str, artifact: ndarray):
        if self._run.is_active:
            self._export_array(dir_path=path, array=artifact)
        else:
            raise InactiveFilesystemRunError("Run is inactive")

    @classmethod
    def _export_array(cls, dir_path: str, array: np.ndarray):
        filepath = IncrementalPathGenerator.generate(dir_path=dir_path, fmt=cls._fmt)
        np.save(file=filepath, arr=array)

    @classmethod
    def _get_relative_path(cls, artifact_name: str) -> str:
        return os.path.join("arrays", artifact_name)
