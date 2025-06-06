import os
import tempfile

import numpy as np

from artifact_experiment.libs.tracking.clear_ml.adapter import ClearMLRunAdapter
from artifact_experiment.libs.tracking.clear_ml.loggers.base import ClearMLArtifactLogger
from artifact_experiment.libs.utils.incremental_path_generator import IncrementalPathGenerator


class ClearMLArrayLogger(ClearMLArtifactLogger[np.ndarray]):
    _fmt = ".npy"

    def _append(self, artifact_path: str, artifact: np.ndarray):
        iteration = self._get_array_iteration(run=self._run, path=artifact_path)
        artifact_name = IncrementalPathGenerator.format_path(
            dir_path=artifact_path, next_idx=iteration
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            local_filepath = IncrementalPathGenerator.format_path(
                dir_path=temp_dir, next_idx=iteration, fmt=self._fmt
            )
            np.save(file=local_filepath, arr=artifact)
            self._run.upload(
                dir_target=artifact_name, path_source=local_filepath, delete_after_upload=False
            )

    @classmethod
    def _get_relative_path(cls, artifact_name: str) -> str:
        return os.path.join("arrays", artifact_name)

    @staticmethod
    def _get_array_iteration(run: ClearMLRunAdapter, path: str) -> int:
        file_store = run.get_exported_files()
        iteration = file_store.get_n_files(path=path)
        return iteration
