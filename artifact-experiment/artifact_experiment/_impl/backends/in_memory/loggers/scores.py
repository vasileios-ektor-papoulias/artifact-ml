from artifact_core.typing import Score

from artifact_experiment._impl.backends.in_memory.loggers.artifacts import InMemoryArtifactLogger


class InMemoryScoreLogger(InMemoryArtifactLogger[Score]):
    def _append(self, item_path: str, item: Score):
        step = 1 + len(self._run.search_score_store(store_path=item_path))
        key = self._get_store_key(item_path=item_path, step=step)
        self._run.log_score(path=key, score=item)

    @classmethod
    def _get_relative_path(cls, item_name: str) -> str:
        return f"scores/{item_name}"
