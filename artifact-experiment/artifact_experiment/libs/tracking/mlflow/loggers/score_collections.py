from typing import Dict

from artifact_experiment.libs.tracking.mlflow.backend import InactiveMlflowRunError
from artifact_experiment.libs.tracking.mlflow.loggers.base import MlflowArtifactLogger


class MlflowScoreCollectionLogger(MlflowArtifactLogger[Dict[str, float]]):
    def _log(self, path: str, artifact: Dict[str, float]):
        if not self._backend.run_is_active:
            raise InactiveMlflowRunError("No active run.")
        client = self._backend.native_client.client
        run_id = self._backend.experiment_id
        for score_name, score_value in artifact.items():
            metric_key = f"{path}/{score_name}"
            history = client.get_metric_history(run_id=run_id, key=metric_key)
            next_step = len(history) + 1
            client.log_metric(run_id=run_id, key=metric_key, value=score_value, step=next_step)

    @classmethod
    def _get_relative_path(cls, artifact_name: str) -> str:
        return f"score_collections/{artifact_name}"
