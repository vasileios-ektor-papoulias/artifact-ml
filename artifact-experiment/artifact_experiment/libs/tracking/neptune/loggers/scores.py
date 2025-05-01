from artifact_experiment.libs.tracking.neptune.backend import (
    NeptuneBackend,
    NoActiveNeptuneRunError,
)
from artifact_experiment.libs.tracking.neptune.loggers.base import NeptuneArtifactLogger


class NeptuneScoreLogger(NeptuneArtifactLogger[float, NeptuneBackend]):
    def _log(self, path: str, artifact: float):
        if self._backend.run_is_active:
            self._backend.native_client[path] = artifact
        else:
            raise NoActiveNeptuneRunError("No active run.")

    @classmethod
    def _get_relative_path(cls, artifact_name: str) -> str:
        return f"scores/{artifact_name}"
