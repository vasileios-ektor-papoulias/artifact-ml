import os
from pathlib import Path

import numpy as np
import pandas as pd
from artifact_core.typing import Score

from artifact_experiment._impl.backends.filesystem.adapter import InactiveFilesystemRunError
from artifact_experiment._impl.backends.filesystem.loggers.artifacts import FilesystemArtifactLogger


class FilesystemScoreLogger(FilesystemArtifactLogger[Score]):
    _column_name: str = "value"

    def _append(self, item_path: str, item: Score):
        if self._run.is_active:
            self._export_score(path=Path(item_path), value=item, column_name=self._column_name)
        else:
            raise InactiveFilesystemRunError("Run is inactive")

    @staticmethod
    def _export_score(path: Path, value: Score, column_name: str):
        os.makedirs(name=path.parent, exist_ok=True)
        if path.exists():
            df = pd.read_csv(path)
        else:
            df = pd.DataFrame(columns=[column_name], dtype=np.float64)
        new_row = pd.DataFrame({column_name: [value]})
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(path, index=False)

    @classmethod
    def _get_relative_path(cls, item_name: str) -> str:
        return os.path.join("scores", item_name)
