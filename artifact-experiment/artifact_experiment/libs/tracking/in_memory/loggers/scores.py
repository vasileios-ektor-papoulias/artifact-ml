from artifact_experiment.libs.tracking.in_memory.loggers.base import (
    InMemoryArtifactLogger,
)


class InMemoryScoreLogger(InMemoryArtifactLogger[float]):
    def _append(self, artifact_path: str, artifact: float):
        step = 1 + len(self._run.search_score_store(artifact_path=artifact_path))
        key = self._get_store_key(artifact_path=artifact_path, step=step)
        self._run.log_score(path=key, score=artifact)

    @classmethod
    def _get_relative_path(cls, artifact_name: str) -> str:
        return f"scores/{artifact_name}"
