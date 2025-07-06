from typing import Dict

from artifact_experiment.libs.tracking.in_memory.loggers.base import (
    InMemoryArtifactLogger,
)


class InMemoryScoreCollectionLogger(InMemoryArtifactLogger[Dict[str, float]]):
    def _append(self, artifact_path: str, artifact: Dict[str, float]):
        step = self._run.n_score_collections + 1
        key = self._get_store_key(artifact_path, step)
        self._run._native_run.dict_score_collections[key] = artifact

    @classmethod
    def _get_relative_path(cls, artifact_name: str) -> str:
        return f"score_collections/{artifact_name}"
