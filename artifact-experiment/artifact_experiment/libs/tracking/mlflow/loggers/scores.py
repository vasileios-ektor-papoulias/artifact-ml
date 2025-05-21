import os
from typing import List

from mlflow.entities import Metric

from artifact_experiment.libs.tracking.mlflow.loggers.base import MlflowArtifactLogger


class MlflowScoreLogger(MlflowArtifactLogger[float]):
    def _log(self, path: str, artifact: float):
        ls_history = self._run.get_ls_score_history(backend_path=path)
        next_step = self._get_next_step_from_history(ls_history=ls_history)
        self._run.log_score(backend_path=path, value=artifact, step=next_step)

    @staticmethod
    def _get_next_step_from_history(ls_history: List[Metric]) -> int:
        return 1 + len(ls_history)

    @classmethod
    def _get_relative_path(cls, artifact_name: str) -> str:
        return os.path.join("scores", artifact_name)
