import os
import tempfile
from typing import Dict

import numpy as np

from artifact_experiment.libs.tracking.mlflow.loggers.base import MlflowArtifactLogger
from artifact_experiment.libs.utils.filesystem import IncrementalPathGenerator


class MlflowArrayCollectionLogger(MlflowArtifactLogger[Dict[str, np.ndarray]]):
    _fmt = "npz"

    def _log(self, path: str, artifact: Dict[str, np.ndarray]):
        ls_existing_filepaths = [
            str(info.path) for info in self._run.get_ls_artifact_info(backend_path=path)
        ]
        with tempfile.TemporaryDirectory() as td:
            local_path = IncrementalPathGenerator.generate_from_existing_filepaths(
                ls_existing_filepaths=ls_existing_filepaths,
                dir_local=td,
                fmt=self._fmt,
            )
            np.savez_compressed(file=local_path, allow_pickle=True, **artifact)
            self._run.upload(
                backend_path=path,
                local_path=local_path,
            )

    @classmethod
    def _get_relative_path(cls, artifact_name: str) -> str:
        return os.path.join("array_collections", artifact_name)
