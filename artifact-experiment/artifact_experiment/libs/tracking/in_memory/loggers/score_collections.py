from typing import Dict

from artifact_experiment.libs.tracking.in_memory.loggers.base import (
    InMemoryArtifactLogger,
)


class InMemoryScoreCollectionLogger(InMemoryArtifactLogger[Dict[str, float]]):
    def _append(self, artifact_path: str, artifact: Dict[str, float]):
        step = 1 + len(self._run.search_score_collection_store(artifact_path=artifact_path))
        key = self._get_store_key(artifact_path, step)
        self._run.log_score_collection(path=key, score_collection=artifact)

    @classmethod
    def _get_relative_path(cls, artifact_name: str) -> str:
        return f"score_collections/{artifact_name}"
