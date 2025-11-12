import os
import tempfile

import numpy as np

from artifact_experiment.libs.tracking.clear_ml.adapter import ClearMLRunAdapter
from artifact_experiment.libs.tracking.clear_ml.loggers.artifacts import ClearMLArtifactLogger
from artifact_experiment.libs.utils.incremental_path_generator import IncrementalPathGenerator


class ClearMLArrayLogger(ClearMLArtifactLogger[Array]):
    _fmt = ".npy"

    def _append(self, item_path: str, item: Array):
        iteration = self._get_array_iteration(run=self._run, path=item_path)
        item_name = IncrementalPathGenerator.format_path(dir_path=item_path, next_idx=iteration)
        with tempfile.TemporaryDirectory() as temp_dir:
            local_filepath = IncrementalPathGenerator.format_path(
                dir_path=temp_dir, next_idx=iteration, fmt=self._fmt
            )
            np.save(file=local_filepath, arr=item)
            self._run.upload(
                dir_target=item_name, path_source=local_filepath, delete_after_upload=False
            )

    @classmethod
    def _get_relative_path(cls, item_name: str) -> str:
        return os.path.join("arrays", item_name)

    @staticmethod
    def _get_array_iteration(run: ClearMLRunAdapter, path: str) -> int:
        file_store = run.get_exported_files()
        iteration = file_store.get_n_files(path=path)
        return iteration
