import os

from artifact_experiment.libs.tracking.mlflow.loggers.base import MlflowArtifactLogger


class MlflowScoreLogger(MlflowArtifactLogger[float]):
    def _log(self, path: str, artifact: float):
        next_step = self._get_next_step_count(run=self._run, path=path)
        self._run.log_score(backend_path=path, value=artifact, step=next_step)

    @classmethod
    def _get_relative_path(cls, artifact_name: str) -> str:
        return os.path.join("scores", artifact_name)
