import os
import tempfile
from pathlib import Path

import numpy as np

from artifact_experiment.libs.tracking.clear_ml.adapter import ClearMLRunAdapter
from artifact_experiment.libs.tracking.clear_ml.loggers.base import ClearMLArtifactLogger


class ClearMLArrayLogger(ClearMLArtifactLogger[np.ndarray]):
    _fmt = ".npy"

    def _log(self, path: str, artifact: np.ndarray):
        iteration = self._get_array_iteration(run=self._run, path=path)
        local_dirpath = self._get_local_dirpath(path=path)
        local_filepath = self._append_iteration(dirpath=local_dirpath, iteration=iteration)
        np.save(file=local_filepath, arr=artifact)
        artifact_name = self._append_iteration(dirpath=path, iteration=iteration)
        self._run.upload(name=artifact_name, filepath=local_filepath, delete_after_upload=True)

    @classmethod
    def _append_iteration(cls, dirpath: str, iteration: int):
        return os.path.join(dirpath, f"{str(iteration)}.{cls._fmt}")

    @staticmethod
    def _get_local_dirpath(path: str) -> str:
        tmp_dir = Path(tempfile.gettempdir())
        local_dirpath = str(tmp_dir / path)
        os.makedirs(name=local_dirpath, exist_ok=True)
        return local_dirpath

    @classmethod
    def _get_relative_path(cls, artifact_name: str) -> str:
        return os.path.join("arrays", artifact_name)

    @staticmethod
    def _get_array_iteration(run: ClearMLRunAdapter, path: str) -> int:
        dict_all_artifacts = run.get_uploaded_files()
        dict_array_history = {
            name: artifact for name, artifact in dict_all_artifacts.items() if path in name
        }
        iteration = len(dict_array_history)
        return iteration
