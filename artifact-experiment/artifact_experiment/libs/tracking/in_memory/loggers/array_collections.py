from typing import Dict

import numpy as np

from artifact_experiment.libs.tracking.in_memory.loggers.base import (
    InMemoryArtifactLogger,
)


class InMemoryArrayCollectionLogger(InMemoryArtifactLogger[Dict[str, np.ndarray]]):
    def _append(self, artifact_path: str, artifact: Dict[str, np.ndarray]):
        step = self._run.n_array_collections + 1
        key = self._get_store_key(artifact_path, step)
        self._run._native_run.dict_array_collections[key] = artifact

    @classmethod
    def _get_relative_path(cls, artifact_name: str) -> str:
        return f"array_collections/{artifact_name}"
