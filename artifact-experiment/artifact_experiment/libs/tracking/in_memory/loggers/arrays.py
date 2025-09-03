import numpy as np

from artifact_experiment.libs.tracking.in_memory.loggers.base import (
    InMemoryArtifactLogger,
)


class InMemoryArrayLogger(InMemoryArtifactLogger[np.ndarray]):
    def _append(self, artifact_path: str, artifact: np.ndarray):
        step = 1 + len(self._run.search_array_store(artifact_path=artifact_path))
        key = self._get_store_key(artifact_path=artifact_path, step=step)
        self._run.log_array(path=key, array=artifact)

    @classmethod
    def _get_relative_path(cls, artifact_name: str) -> str:
        return f"arrays/{artifact_name}"
