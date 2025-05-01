import os
import tempfile

import numpy as np

from artifact_experiment.libs.tracking.mlflow.backend import InactiveMlflowRunError
from artifact_experiment.libs.tracking.mlflow.loggers.base import MlflowArtifactLogger
from artifact_experiment.libs.utils.filesystem import IncrementalPathGenerator


class MlflowArrayLogger(MlflowArtifactLogger[np.ndarray]):
    _fmt = "npy"

    def _log(self, path: str, artifact: np.ndarray):
        if not self._backend.run_is_active:
            raise InactiveMlflowRunError("No active experiment.")
        run_id = self._backend.experiment_id
        artifact_dir = os.path.dirname(path)
        with tempfile.TemporaryDirectory() as td:
            local_path = IncrementalPathGenerator.generate_mlflow(
                client=self._backend.native_client.client,
                run_id=run_id,
                remote_path=artifact_dir,
                dir_local=td,
                fmt=self._fmt,
            )
            np.save(file=local_path, arr=artifact)
            self._backend.native_client.client.log_artifact(
                run_id=run_id,
                local_path=local_path,
                artifact_path=artifact_dir,
            )
