import os
from typing import Dict

from artifact_experiment.libs.tracking.mlflow.loggers.base import MlflowArtifactLogger


class MlflowScoreCollectionLogger(MlflowArtifactLogger[Dict[str, float]]):
    def _log(self, path: str, artifact: Dict[str, float]):
        for score_name, score_value in artifact.items():
            backend_path = f"{path}/{score_name}"
            history = self._run.get_ls_score_history(backend_path=backend_path)
            next_step = len(history) + 1
            self._run.log_score(backend_path=backend_path, value=score_value, step=next_step)

    @classmethod
    def _get_relative_path(cls, artifact_name: str) -> str:
        return os.path.join("score_collections", artifact_name)
