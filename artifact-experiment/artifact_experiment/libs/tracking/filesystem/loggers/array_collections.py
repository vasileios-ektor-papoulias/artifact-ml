import os
from typing import Dict

import numpy as np

from artifact_experiment.libs.tracking.filesystem.adapter import (
    InactiveFilesystemRunError,
)
from artifact_experiment.libs.tracking.filesystem.loggers.base import FilesystemArtifactLogger
from artifact_experiment.libs.utils.incremental_path_generator import (
    IncrementalPathGenerator,
)


class FilesystemArrayCollectionLogger(FilesystemArtifactLogger[Dict[str, np.ndarray]]):
    _fmt: str = "npz"

    def _append(self, artifact_path: str, artifact: Dict[str, np.ndarray]):
        if self._run.is_active:
            self._export_array_collection(dir_path=artifact_path, array_collection=artifact)
        else:
            raise InactiveFilesystemRunError("Run is inactive")

    @classmethod
    def _export_array_collection(cls, dir_path: str, array_collection: Dict[str, np.ndarray]):
        filepath = IncrementalPathGenerator.generate(dir_path=dir_path, fmt=cls._fmt)
        np.savez_compressed(file=filepath, allow_pickle=True, **array_collection)

    @classmethod
    def _get_relative_path(cls, artifact_name: str) -> str:
        return os.path.join("array_collections", artifact_name)
