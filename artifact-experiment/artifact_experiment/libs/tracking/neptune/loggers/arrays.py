from numpy import ndarray

from artifact_experiment.libs.tracking.neptune.backend import (
    NoActiveNeptuneRunError,
)
from artifact_experiment.libs.tracking.neptune.loggers.base import NeptuneArtifactLogger


class NeptuneArrayLogger(NeptuneArtifactLogger[ndarray]):
    def _log(self, path: str, artifact: ndarray):
        if self._backend.run_is_active:
            self._backend.native_client[path] = artifact
        else:
            raise NoActiveNeptuneRunError("No active run.")

    @classmethod
    def _get_relative_path(cls, artifact_name: str) -> str:
        return f"arrays/{artifact_name}"
