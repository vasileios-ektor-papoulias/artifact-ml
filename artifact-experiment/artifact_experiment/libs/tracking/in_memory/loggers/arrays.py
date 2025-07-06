import numpy as np

from artifact_experiment.libs.tracking.in_memory.loggers.base import (
    InMemoryArtifactLogger,
)


class InMemoryArrayLogger(InMemoryArtifactLogger[np.ndarray]):
    def _append(self, artifact_path: str, artifact: np.ndarray):
        step = self._run.n_arrays + 1
        key = self._get_store_key(artifact_path, step)
        self._run._native_run.dict_arrays[key] = artifact

    @classmethod
    def _get_relative_path(cls, artifact_name: str) -> str:
        return f"arrays/{artifact_name}"
