from numpy import ndarray

from artifact_experiment.base.tracking.logger import ArtifactLogger
from artifact_experiment.tracking.neptune.backend import (
    NeptuneBackend,
    NeptuneExperimentNotSetError,
)


class NeptuneArrayLogger(ArtifactLogger[ndarray, NeptuneBackend]):
    def __init__(self, backend: NeptuneBackend):
        self._backend = backend

    def _log(self, path: str, artifact: ndarray):
        if self._backend.experiment_is_active:
            self._backend.native_client[path] = artifact
        else:
            raise NeptuneExperimentNotSetError("No active experiment.")

    @classmethod
    def _get_relative_path(cls, artifact_name: str) -> str:
        return f"arrays/{artifact_name}"
