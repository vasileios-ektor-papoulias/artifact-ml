from typing import Dict

from artifact_experiment.libs.tracking.neptune.backend import (
    NoActiveNeptuneRunError,
)
from artifact_experiment.libs.tracking.neptune.loggers.base import NeptuneArtifactLogger


class NeptuneScoreCollectionLogger(NeptuneArtifactLogger[Dict[str, float]]):
    def _log(self, path: str, artifact: Dict[str, float]):
        if self._backend.run_is_active:
            self._backend.native_client[path] = artifact
        else:
            raise NoActiveNeptuneRunError("No active run.")

    @classmethod
    def _get_relative_path(cls, artifact_name: str) -> str:
        return f"score_collections/{artifact_name}"
