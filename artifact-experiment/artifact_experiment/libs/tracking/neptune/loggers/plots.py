from matplotlib.figure import Figure

from artifact_experiment.libs.tracking.neptune.backend import (
    NeptuneBackend,
    NoActiveNeptuneRunError,
)
from artifact_experiment.libs.tracking.neptune.loggers.base import NeptuneArtifactLogger


class NeptunePlotLogger(NeptuneArtifactLogger[Figure, NeptuneBackend]):
    def _log(self, path: str, artifact: Figure):
        if self._backend.run_is_active:
            self._backend.native_client[path] = artifact
        else:
            raise NoActiveNeptuneRunError("No active run.")

    @classmethod
    def _get_relative_path(cls, artifact_name: str) -> str:
        return f"plots/{artifact_name}"
