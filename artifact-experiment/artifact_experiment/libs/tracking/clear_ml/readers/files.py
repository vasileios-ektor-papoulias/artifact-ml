from typing import Dict

from clearml.binding.artifacts import Artifact

from artifact_experiment.libs.tracking.clear_ml.adapter import (
    ClearMLRunAdapter,
)
from artifact_experiment.libs.tracking.clear_ml.readers.base import ClearMLReader


class ClearMLFileReader(ClearMLReader):
    @classmethod
    def get_file_history(cls, run: ClearMLRunAdapter, path: str) -> Dict[str, Artifact]:
        path = cls._get_full_path(path=path)
        dict_all_files = cls._get_all_files(run=run)
        dict_file_history = cls._get_file_history(dict_all_files=dict_all_files, remote_path=path)
        return dict_file_history

    @classmethod
    def _get_file_history(
        cls,
        dict_all_files: Dict[str, Artifact],
        remote_path: str,
    ) -> Dict[str, Artifact]:
        dict_file_history = {
            name: file for name, file in dict_all_files.items() if remote_path in name
        }
        return dict_file_history

    @classmethod
    def _get_all_files(cls, run: ClearMLRunAdapter) -> Dict[str, Artifact]:
        dict_all_files = run.get_uploaded_files()
        return dict_all_files
