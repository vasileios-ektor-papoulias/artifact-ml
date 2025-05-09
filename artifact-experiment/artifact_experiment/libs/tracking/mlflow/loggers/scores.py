import os

from artifact_experiment.libs.tracking.mlflow.loggers.base import MlflowArtifactLogger


class MlflowScoreLogger(MlflowArtifactLogger[float]):
    def _log(self, path: str, artifact: float):
        history = self._run.get_ls_score_history(backend_path=path)
        next_step = len(history) + 1
        self._run.log_score(backend_path=path, value=artifact, step=next_step)

    @classmethod
    def _get_relative_path(cls, artifact_name: str) -> str:
        return os.path.join("scores", artifact_name)
