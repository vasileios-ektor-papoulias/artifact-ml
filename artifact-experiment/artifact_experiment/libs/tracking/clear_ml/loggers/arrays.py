import os
import tempfile
from pathlib import Path

import numpy as np

from artifact_experiment.libs.tracking.clear_ml.loggers.base import ClearMLArtifactLogger


class ClearMLArrayLogger(ClearMLArtifactLogger[np.ndarray]):
    _fmt = ".npy"

    def _log(self, path: str, artifact: np.ndarray):
        local_dirpath = self._get_local_dirpath(path=path)
        local_filepath = self._get_local_filepath(dirpath=local_dirpath, iteration=self._iteration)
        np.save(file=local_filepath, arr=artifact)
        self._run.upload_artifact(name=path, filepath=local_filepath, delete_after_upload=True)
        self._iteration += 1

    @classmethod
    def _get_local_filepath(cls, dirpath: str, iteration: int):
        return os.path.join(dirpath, f"{str(iteration)}.{cls._fmt}")

    @staticmethod
    def _get_local_dirpath(path: str) -> str:
        tmp_dir = Path(tempfile.gettempdir())
        local_dirpath = str(tmp_dir / path)
        os.makedirs(name=local_dirpath, exist_ok=True)
        return local_dirpath

    @classmethod
    def _get_relative_path(cls, artifact_name: str) -> str:
        return f"arrays/{artifact_name}"
