from pathlib import Path
from typing import Dict

import pandas as pd

from artifact_experiment.libs.tracking.filesystem.adapter import (
    NoActiveFilesystemRunError,
)
from artifact_experiment.libs.tracking.filesystem.loggers.base import FilesystemArtifactLogger


class FilesystemScoreCollectionLogger(FilesystemArtifactLogger[Dict[str, float]]):
    def _log(self, path: str, artifact: Dict[str, float]):
        if self._run.run_is_active:
            self._export_score_collection(path=Path(path), dict_values=artifact)
        else:
            raise NoActiveFilesystemRunError("No active run.")

    @staticmethod
    def _export_score_collection(path: Path, dict_values: Dict[str, float]):
        if path.exists():
            df = pd.read_csv(path)
        else:
            df = pd.DataFrame(columns=list(dict_values.keys()))
        new_row = pd.DataFrame([dict_values])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(path, index=False)

    @classmethod
    def _get_relative_path(cls, artifact_name: str) -> str:
        return f"score_collections/{artifact_name}"
