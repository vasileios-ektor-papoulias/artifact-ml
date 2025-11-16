from artifact_core.typing import ScoreCollection

from artifact_experiment._impl.backends.in_memory.loggers.artifacts import InMemoryArtifactLogger


class InMemoryScoreCollectionLogger(InMemoryArtifactLogger[ScoreCollection]):
    def _append(self, item_path: str, item: ScoreCollection):
        step = 1 + len(self._run.search_score_collection_store(store_path=item_path))
        key = self._get_store_key(item_path=item_path, step=step)
        self._run.log_score_collection(path=key, score_collection=item)

    @classmethod
    def _get_relative_path(cls, item_name: str) -> str:
        return f"score_collections/{item_name}"
