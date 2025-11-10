from artifact_experiment.libs.tracking.in_memory.loggers.artifacts import InMemoryArtifactLogger


class InMemoryScoreLogger(InMemoryArtifactLogger[float]):
    def _append(self, item_path: str, item: float):
        step = 1 + len(self._run.search_score_store(artifact_path=item_path))
        key = self._get_store_key(item_path=item_path, step=step)
        self._run.log_score(path=key, score=item)

    @classmethod
    def _get_relative_path(cls, item_name: str) -> str:
        return f"scores/{item_name}"
