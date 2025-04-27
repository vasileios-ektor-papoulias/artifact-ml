from typing import Dict

import numpy as np

from artifact_experiment.base.tracking.logger import ArtifactLogger
from artifact_experiment.libs.utils.filesystem import (
    IncrementalPathGenerator,
)
from artifact_experiment.tracking.filesystem.backend import (
    FilesystemBackend,
    FilesystemExperimentNotSetError,
)


class FilesystemArrayCollectionLogger(ArtifactLogger[Dict[str, np.ndarray], FilesystemBackend]):
    def __init__(self, backend: FilesystemBackend):
        self._backend = backend

    def _log(self, path: str, artifact: Dict[str, np.ndarray]):
        if self._backend.experiment_is_active:
            self._export_array_collection(dir_path=path, array_collection=artifact)
        else:
            raise FilesystemExperimentNotSetError("No active experiment.")

    @classmethod
    def _export_array_collection(cls, dir_path: str, array_collection: Dict[str, np.ndarray]):
        filepath = IncrementalPathGenerator.generate(dir_path=dir_path)
        np.savez_compressed(file=filepath, allow_pickle=True, **array_collection)

    @classmethod
    def _get_relative_path(cls, artifact_name: str) -> str:
        return f"array_collections/{artifact_name}"
