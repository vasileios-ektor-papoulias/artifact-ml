import os
import tempfile
from typing import Dict, List

import numpy as np
from mlflow.entities import FileInfo

from artifact_experiment.libs.tracking.mlflow.loggers.base import (
    MlflowArtifactLogger,
    MlflowRunAdapter,
)
from artifact_experiment.libs.utils.filesystem import IncrementalPathGenerator


class MlflowArrayCollectionLogger(MlflowArtifactLogger[Dict[str, np.ndarray]]):
    _fmt = "npz"

    def _append(self, artifact_path: str, artifact: Dict[str, np.ndarray]):
        ls_history = self._get_array_collection_history(run=self._run, path=artifact_path)
        next_step = self._get_next_step_from_history(ls_history=ls_history)
        with tempfile.TemporaryDirectory() as temp_dir:
            local_path = IncrementalPathGenerator.format_path(
                dir_path=temp_dir,
                next_idx=next_step,
                fmt=self._fmt,
            )
            np.savez_compressed(file=local_path, allow_pickle=True, **artifact)
            self._run.upload(path_source=local_path, dir_target=artifact_path)

    @staticmethod
    def _get_array_collection_history(run: MlflowRunAdapter, path: str) -> List[FileInfo]:
        ls_history = run.get_ls_artifact_info(backend_path=path)
        return ls_history

    @staticmethod
    def _get_next_step_from_history(ls_history: List[FileInfo]) -> int:
        return 1 + len(ls_history)

    @classmethod
    def _get_relative_path(cls, artifact_name: str) -> str:
        return os.path.join("array_collections", artifact_name)
