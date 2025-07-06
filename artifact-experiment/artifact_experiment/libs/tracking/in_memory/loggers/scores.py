from artifact_experiment.libs.tracking.in_memory.loggers.base import (
    InMemoryArtifactLogger,
)


class InMemoryScoreLogger(InMemoryArtifactLogger[float]):
    def _append(self, artifact_path: str, artifact: float):
        step = self._run.n_scores + 1
        key = self._get_store_key(artifact_path, step)
        self._run._native_run.dict_scores[key] = artifact

    @classmethod
    def _get_relative_path(cls, artifact_name: str) -> str:
        return f"scores/{artifact_name}"
