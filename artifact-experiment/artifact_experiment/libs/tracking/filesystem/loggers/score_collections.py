from pathlib import Path
from typing import Dict

import pandas as pd

from artifact_experiment.base.tracking.logger import ArtifactLogger
from artifact_experiment.libs.tracking.filesystem.backend import (
    FilesystemBackend,
    FilesystemExperimentNotSetError,
)


class FilesystemScoreCollectionLogger(ArtifactLogger[Dict[str, float], FilesystemBackend]):
    def __init__(self, backend: FilesystemBackend):
        self._backend = backend

    def _log(self, path: str, artifact: Dict[str, float]):
        if self._backend.native_client is not None and self._backend.experiment_is_active:
            self._export_score_collection(path=Path(path), dict_values=artifact)
        else:
            raise FilesystemExperimentNotSetError("No active experiment.")

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
