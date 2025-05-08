from artifact_experiment.libs.tracking.clear_ml.loggers.base import ClearMLArtifactLogger


class ClearMLScoreLogger(ClearMLArtifactLogger[float]):
    def _log(self, path: str, artifact: float):
        self._run.log_score(value=artifact, title=path, iteration=self._iteration)
        self._iteration += 1

    @classmethod
    def _get_relative_path(cls, artifact_name: str) -> str:
        return f"scores/{artifact_name}"
