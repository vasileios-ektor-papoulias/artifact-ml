import os
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from artifact_experiment.libs.tracking.filesystem.adapter import (
    InactiveFilesystemRunError,
)
from artifact_experiment.libs.tracking.filesystem.loggers.base import FilesystemArtifactLogger


class FilesystemScoreCollectionLogger(FilesystemArtifactLogger[Dict[str, float]]):
    def _append(self, artifact_path: str, artifact: Dict[str, float]):
        if self._run.is_active:
            self._export_score_collection(path=Path(artifact_path), dict_values=artifact)
        else:
            raise InactiveFilesystemRunError("Run is inactive")

    @staticmethod
    def _export_score_collection(path: Path, dict_values: Dict[str, float]):
        os.makedirs(name=path.parent, exist_ok=True)
        if path.exists():
            df = pd.read_csv(path)
        else:
            df = pd.DataFrame(columns=list(dict_values.keys()), dtype=np.float64)
        new_row = pd.DataFrame([dict_values])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(path, index=False)

    @classmethod
    def _get_relative_path(cls, artifact_name: str) -> str:
        return os.path.join("score_collections", artifact_name)
