from mlflow.tracking import MlflowClient

from artifact_experiment.libs.tracking.mlflow.backend import InactiveMlflowRunError
from artifact_experiment.libs.tracking.mlflow.loggers.base import MlflowArtifactLogger


class MlflowScoreLogger(MlflowArtifactLogger[float]):
    def _log(self, path: str, artifact: float):
        if not self._backend.run_is_active:
            raise InactiveMlflowRunError("No active run.")
        client: MlflowClient = self._backend.native_client.client
        run_id = self._backend.experiment_id
        history = client.get_metric_history(run_id=run_id, key=path)
        next_step = len(history) + 1
        client.log_metric(run_id=run_id, key=path, value=artifact, step=next_step)

    @classmethod
    def _get_relative_path(cls, artifact_name: str) -> str:
        return f"scores/{artifact_name}"
