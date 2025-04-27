from artifact_experiment.base.tracking.logger import ArtifactLogger
from artifact_experiment.tracking.neptune.backend import (
    NeptuneBackend,
    NeptuneExperimentNotSetError,
)


class NeptuneScoreLogger(ArtifactLogger[float, NeptuneBackend]):
    def __init__(self, backend: NeptuneBackend):
        self._backend = backend

    def _log(self, path: str, artifact: float):
        if self._backend.native_client is not None and self._backend.experiment_is_active:
            self._backend.native_client[path] = artifact
        else:
            raise NeptuneExperimentNotSetError("No active experiment.")

    @classmethod
    def _get_relative_path(cls, artifact_name: str) -> str:
        return "scores/artifact_name"
