from typing import Dict

from clearml.binding.artifacts import Artifact

from artifact_experiment.libs.tracking.clear_ml.adapter import (
    ClearMLRunAdapter,
)


class ClearMLFileReader:
    @classmethod
    def get_all_files(cls, run: ClearMLRunAdapter) -> Dict[str, Artifact]:
        dict_all_files = run.get_uploaded_files()
        return dict_all_files

    @classmethod
    def get_file_history(
        cls,
        dict_all_files: Dict[str, Artifact],
        remote_path: str,
    ) -> Dict[str, Artifact]:
        dict_file_history = {
            name: file for name, file in dict_all_files.items() if remote_path in name
        }
        return dict_file_history
