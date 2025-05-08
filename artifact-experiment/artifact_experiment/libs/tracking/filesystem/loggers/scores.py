import os
from pathlib import Path

import pandas as pd

from artifact_experiment.libs.tracking.filesystem.adapter import (
    NoActiveFilesystemRunError,
)
from artifact_experiment.libs.tracking.filesystem.loggers.base import FilesystemArtifactLogger


class FilesystemScoreLogger(FilesystemArtifactLogger[float]):
    _column_name: str = "value"

    def _log(self, path: str, artifact: float):
        if self._run.run_is_active:
            self._export_score(path=Path(path), value=artifact, column_name=self._column_name)
        else:
            raise NoActiveFilesystemRunError("No active run.")

    @staticmethod
    def _export_score(path: Path, value: float, column_name: str):
        if path.exists():
            df = pd.read_csv(path)
        else:
            df = pd.DataFrame(columns=[column_name])
        new_row = pd.DataFrame({column_name: [value]})
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(path, index=False)

    @classmethod
    def _get_relative_path(cls, artifact_name: str) -> str:
        relative_dirpath = "scores"
        os.makedirs(name=relative_dirpath, exist_ok=True)
        return f"{relative_dirpath}/{artifact_name}"
