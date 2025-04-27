from pathlib import Path

import pandas as pd

from artifact_experiment.base.tracking.logger import ArtifactLogger
from artifact_experiment.tracking.filesystem.backend import (
    FilesystemBackend,
    FilesystemExperimentNotSetError,
)


class FilesystemScoreCollectionLogger(ArtifactLogger[float, FilesystemBackend]):
    _column_name: str = "value"

    def __init__(self, backend: FilesystemBackend):
        self._backend = backend

    def _log(self, path: str, artifact: float):
        if self._backend.native_client is not None and self._backend.experiment_is_active:
            self._export_score(path=Path(path), value=artifact, column_name=self._column_name)
        else:
            raise FilesystemExperimentNotSetError("No active experiment.")

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
        return f"scores/{artifact_name}"
