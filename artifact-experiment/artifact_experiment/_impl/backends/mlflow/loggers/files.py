from typing import List

from mlflow.entities import FileInfo

from artifact_experiment._base.primitives.file import File
from artifact_experiment._impl.backends.mlflow.adapter import MlflowRunAdapter
from artifact_experiment._impl.backends.mlflow.loggers.base import MlflowLogger
from artifact_experiment._utils.filesystem.incremental_paths import IncrementalPathFormatter


class MlflowFileLogger(MlflowLogger[File]):
    _root_dir = "files"

    def _append(self, item_path: str, item: File):
        ls_history = self._get_file_history(run=self._run, path=item_path)
        next_step = self._get_next_step_from_history(ls_history=ls_history)
        backend_dir = IncrementalPathFormatter.format(dir_path=item_path, next_idx=next_step)
        self._run.log_file(backend_dir=backend_dir, file=item)

    @staticmethod
    def _get_file_history(run: MlflowRunAdapter, path: str) -> List[FileInfo]:
        ls_history = run.get_ls_artifact_info(backend_path=path)
        return ls_history

    @staticmethod
    def _get_next_step_from_history(ls_history: List[FileInfo]) -> int:
        return 1 + len(ls_history)

    @classmethod
    def _get_relative_path(cls, item_name: str) -> str:
        return item_name

    def _get_root_dir(self) -> str:
        return self._root_dir
