import os
from typing import Dict

import numpy as np

from artifact_experiment.libs.tracking.filesystem.adapter import InactiveFilesystemRunError
from artifact_experiment.libs.tracking.filesystem.loggers.artifacts import FilesystemArtifactLogger
from artifact_experiment.libs.utils.incremental_path_generator import IncrementalPathGenerator


class FilesystemArrayCollectionLogger(FilesystemArtifactLogger[Dict[str, np.ndarray]]):
    _fmt: str = "npz"

    def _append(self, item_path: str, item: Dict[str, np.ndarray]):
        if self._run.is_active:
            self._export_array_collection(dir_path=item_path, array_collection=item)
        else:
            raise InactiveFilesystemRunError("Run is inactive")

    @classmethod
    def _export_array_collection(cls, dir_path: str, array_collection: Dict[str, np.ndarray]):
        filepath = IncrementalPathGenerator.generate(dir_path=dir_path, fmt=cls._fmt)
        np.savez_compressed(file=filepath, allow_pickle=True, **array_collection)

    @classmethod
    def _get_relative_path(cls, item_name: str) -> str:
        return os.path.join("array_collections", item_name)
