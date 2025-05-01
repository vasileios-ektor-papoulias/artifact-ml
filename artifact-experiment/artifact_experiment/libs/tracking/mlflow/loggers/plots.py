import os
import tempfile

from matplotlib.figure import Figure

from artifact_experiment.libs.tracking.mlflow.backend import InactiveMlflowRunError
from artifact_experiment.libs.tracking.mlflow.loggers.base import MlflowArtifactLogger
from artifact_experiment.libs.utils.filesystem import IncrementalPathGenerator


class MLFlowPlotLogger(MlflowArtifactLogger[Figure]):
    _fmt = "png"

    def _log(self, path: str, artifact: Figure):
        if not self._backend.run_is_active:
            raise InactiveMlflowRunError("No active experiment.")
        client = self._backend.native_client.client
        run_id = self._backend.experiment_id
        artifact_dir = os.path.dirname(path)
        with tempfile.TemporaryDirectory() as td:
            local_path = IncrementalPathGenerator.generate_mlflow(
                client=client,
                run_id=run_id,
                remote_path=artifact_dir,
                dir_local=td,
                fmt=self._fmt,
            )
            artifact.savefig(fname=local_path, format="png", bbox_inches="tight")
            client.log_artifact(
                run_id=run_id,
                local_path=local_path,
                artifact_path=artifact_dir,
            )

    @classmethod
    def _get_relative_path(cls, artifact_name: str) -> str:
        return f"plots/{artifact_name}"
