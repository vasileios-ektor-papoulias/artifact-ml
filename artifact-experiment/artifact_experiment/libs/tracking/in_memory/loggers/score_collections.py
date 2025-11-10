from typing import Dict

from artifact_experiment.libs.tracking.in_memory.loggers.artifacts import InMemoryArtifactLogger


class InMemoryScoreCollectionLogger(InMemoryArtifactLogger[Dict[str, float]]):
    def _append(self, item_path: str, item: Dict[str, float]):
        step = 1 + len(self._run.search_score_collection_store(artifact_path=item_path))
        key = self._get_store_key(item_path=item_path, step=step)
        self._run.log_score_collection(path=key, score_collection=item)

    @classmethod
    def _get_relative_path(cls, item_name: str) -> str:
        return f"score_collections/{item_name}"
