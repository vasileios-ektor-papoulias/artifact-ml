import os
import tempfile
from typing import List

import numpy as np
from artifact_core.typing import Array
from mlflow.entities import FileInfo

from artifact_experiment._base.primitives.file import File
from artifact_experiment._impl.backends.mlflow.adapter import MlflowRunAdapter
from artifact_experiment._impl.backends.mlflow.loggers.artifacts import MlflowArtifactLogger
from artifact_experiment._utils.filesystem.incremental_paths import IncrementalPathFormatter


class MlflowArrayLogger(MlflowArtifactLogger[Array]):
    _fmt = "npy"

    def _append(self, item_path: str, item: Array):
        ls_history = self._get_array_history(run=self._run, path=item_path)
        next_step = self._get_next_step_from_history(ls_history=ls_history)
        with tempfile.TemporaryDirectory() as temp_dir:
            local_path = IncrementalPathFormatter.format(
                dir_path=temp_dir,
                next_idx=next_step,
                extension=self._fmt,
            )
            np.save(file=local_path, arr=item, allow_pickle=True)
            file = File(path_source=local_path)
            self._run.log_file(backend_dir=item_path, file=file)

    @staticmethod
    def _get_array_history(run: MlflowRunAdapter, path: str) -> List[FileInfo]:
        ls_history = run.get_ls_artifact_info(backend_path=path)
        return ls_history

    @staticmethod
    def _get_next_step_from_history(ls_history: List[FileInfo]) -> int:
        return 1 + len(ls_history)

    @classmethod
    def _get_relative_path(cls, item_name: str) -> str:
        return os.path.join("arrays", item_name)
