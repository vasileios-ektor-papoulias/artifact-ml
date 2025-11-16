import os
import tempfile

import numpy as np
from artifact_core.typing import Array

from artifact_experiment._base.primitives.file import File
from artifact_experiment._impl.backends.clear_ml.adapter import ClearMLRunAdapter
from artifact_experiment._impl.backends.clear_ml.loggers.artifacts import ClearMLArtifactLogger
from artifact_experiment._utils.filesystem.incremental_paths import IncrementalPathFormatter


class ClearMLArrayLogger(ClearMLArtifactLogger[Array]):
    _fmt = ".npy"

    def _append(self, item_path: str, item: Array):
        iteration = self._get_array_iteration(run=self._run, path=item_path)
        item_path = IncrementalPathFormatter.format(dir_path=item_path, next_idx=iteration)
        with tempfile.TemporaryDirectory() as temp_dir:
            local_filepath = IncrementalPathFormatter.format(
                dir_path=temp_dir, next_idx=iteration, extension=self._fmt
            )
            np.save(file=local_filepath, arr=item)
            file = File(path_source=local_filepath)
            self._run.log_file(backend_dir=item_path, file=file, delete_after_upload=False)

    @classmethod
    def _get_relative_path(cls, item_name: str) -> str:
        return os.path.join("arrays", item_name)

    @staticmethod
    def _get_array_iteration(run: ClearMLRunAdapter, path: str) -> int:
        file_store = run.get_exported_files()
        iteration = file_store.get_n_files(path=path)
        return iteration
