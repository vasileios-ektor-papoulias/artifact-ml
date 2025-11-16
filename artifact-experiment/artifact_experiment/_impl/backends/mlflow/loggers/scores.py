import os
from typing import List

from artifact_core.typing import Score
from mlflow.entities import Metric

from artifact_experiment._impl.backends.mlflow.loggers.artifacts import MlflowArtifactLogger


class MlflowScoreLogger(MlflowArtifactLogger[Score]):
    def _append(self, item_path: str, item: Score):
        ls_history = self._run.get_ls_score_history(backend_path=item_path)
        next_step = self._get_next_step_from_history(ls_history=ls_history)
        self._run.log_score(backend_path=item_path, value=item, step=next_step)

    @staticmethod
    def _get_next_step_from_history(ls_history: List[Metric]) -> int:
        return 1 + len(ls_history)

    @classmethod
    def _get_relative_path(cls, item_name: str) -> str:
        return os.path.join("scores", item_name)
