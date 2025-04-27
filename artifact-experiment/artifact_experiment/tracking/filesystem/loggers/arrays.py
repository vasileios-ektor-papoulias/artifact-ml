import numpy as np
from numpy import ndarray

from artifact_experiment.base.tracking.logger import ArtifactLogger
from artifact_experiment.libs.utils.filesystem import (
    IncrementalPathGenerator,
)
from artifact_experiment.tracking.filesystem.backend import (
    FilesystemBackend,
    FilesystemExperimentNotSetError,
)


class FilesystemArrayLogger(ArtifactLogger[ndarray, FilesystemBackend]):
    def __init__(self, backend: FilesystemBackend):
        self._backend = backend

    def _log(self, path: str, artifact: ndarray):
        if self._backend.experiment_is_active:
            self._export_array(dir_path=path, array=artifact)
        else:
            raise FilesystemExperimentNotSetError("No active experiment.")

    @classmethod
    def _export_array(cls, dir_path: str, array: np.ndarray):
        filepath = IncrementalPathGenerator.generate(dir_path=dir_path)
        np.save(file=filepath, arr=array)

    @classmethod
    def _get_relative_path(cls, artifact_name: str) -> str:
        return f"arrays/{artifact_name}"
