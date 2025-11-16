import os

import numpy as np
from artifact_core.typing import ArrayCollection

from artifact_experiment._impl.backends.filesystem.adapter import InactiveFilesystemRunError
from artifact_experiment._impl.backends.filesystem.loggers.artifacts import FilesystemArtifactLogger
from artifact_experiment._utils.filesystem.incremental_paths import IncrementalPathGenerator


class FilesystemArrayCollectionLogger(FilesystemArtifactLogger[ArrayCollection]):
    _fmt: str = "npz"

    def _append(self, item_path: str, item: ArrayCollection):
        if self._run.is_active:
            self._export_array_collection(dir_path=item_path, array_collection=item)
        else:
            raise InactiveFilesystemRunError("Run is inactive")

    @classmethod
    def _export_array_collection(cls, dir_path: str, array_collection: ArrayCollection):
        filepath = IncrementalPathGenerator.generate(dir_path=dir_path, ext=cls._fmt)
        np.savez_compressed(file=filepath, allow_pickle=True, **array_collection)

    @classmethod
    def _get_relative_path(cls, item_name: str) -> str:
        return os.path.join("array_collections", item_name)
