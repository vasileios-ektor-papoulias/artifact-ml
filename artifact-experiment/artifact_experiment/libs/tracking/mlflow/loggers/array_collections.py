import tempfile
from typing import Dict

import numpy as np

from artifact_experiment.libs.tracking.mlflow.backend import (
    InactiveMlflowRunError,
    MlflowBackend,
)
from artifact_experiment.libs.tracking.mlflow.loggers.base import MlflowArtifactLogger
from artifact_experiment.libs.utils.filesystem import IncrementalPathGenerator


class MlflowArrayCollectionLogger(MlflowArtifactLogger[Dict[str, np.ndarray]]):
    _fmt = "npz"

    def __init__(self, backend: MlflowBackend):
        self._backend = backend

    def _log(self, path: str, artifact: Dict[str, np.ndarray]):
        if not self._backend.run_is_active:
            raise InactiveMlflowRunError("No active experiment.")
        run_id = self._backend.experiment_id
        remote_dir = path
        with tempfile.TemporaryDirectory() as td:
            local_path = IncrementalPathGenerator.generate_mlflow(
                client=self._backend.native_client.client,
                run_id=run_id,
                remote_path=remote_dir,
                dir_local=td,
                fmt=self._fmt,
            )
            np.savez_compressed(file=local_path, allow_pickle=True, **artifact)
            self._backend.native_client.client.log_artifact(
                run_id=run_id,
                local_path=local_path,
                artifact_path=remote_dir,
            )

    @classmethod
    def _get_relative_path(cls, artifact_name: str) -> str:
        return f"array_collections/{artifact_name}"
