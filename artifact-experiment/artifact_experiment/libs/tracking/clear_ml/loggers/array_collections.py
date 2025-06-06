import os
import tempfile
from typing import Dict

import numpy as np

from artifact_experiment.libs.tracking.clear_ml.adapter import ClearMLRunAdapter
from artifact_experiment.libs.tracking.clear_ml.loggers.base import ClearMLArtifactLogger
from artifact_experiment.libs.utils.incremental_path_generator import IncrementalPathGenerator


class ClearMLArrayCollectionLogger(ClearMLArtifactLogger[Dict[str, np.ndarray]]):
    _fmt = ".npz"

    def _append(self, artifact_path: str, artifact: Dict[str, np.ndarray]):
        iteration = self._get_array_collection_iteration(run=self._run, path=artifact_path)
        artifact_name = IncrementalPathGenerator.format_path(
            dir_path=artifact_path, next_idx=iteration
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            local_filepath = IncrementalPathGenerator.format_path(
                dir_path=temp_dir, next_idx=iteration, fmt=self._fmt
            )
            np.savez_compressed(file=local_filepath, allow_pickle=True, **artifact)
            self._run.upload(
                dir_target=artifact_name, path_source=local_filepath, delete_after_upload=False
            )

    @classmethod
    def _get_relative_path(cls, artifact_name: str) -> str:
        return os.path.join("array_collections", artifact_name)

    @staticmethod
    def _get_array_collection_iteration(run: ClearMLRunAdapter, path: str) -> int:
        file_store = run.get_exported_files()
        iteration = file_store.get_n_files(path=path)
        return iteration
