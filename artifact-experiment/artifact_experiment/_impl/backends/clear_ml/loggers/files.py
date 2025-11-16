from artifact_experiment._base.primitives.file import File
from artifact_experiment._impl.backends.clear_ml.adapter import ClearMLRunAdapter
from artifact_experiment._impl.backends.clear_ml.loggers.base import ClearMLLogger
from artifact_experiment._utils.filesystem.incremental_paths import IncrementalPathFormatter


class ClearMLFileLogger(ClearMLLogger[File]):
    _root_dir = "files"

    def _append(self, item_path: str, item: File):
        iteration = self._get_file_iteration(run=self._run, path=item_path)
        item_path = IncrementalPathFormatter.format(dir_path=item_path, next_idx=iteration)
        self._run.log_file(backend_dir=item_path, file=item)

    @classmethod
    def _get_relative_path(cls, item_name: str) -> str:
        return item_name

    def _get_root_dir(self) -> str:
        return self._root_dir

    @staticmethod
    def _get_file_iteration(run: ClearMLRunAdapter, path: str) -> int:
        file_store = run.get_exported_files()
        iteration = file_store.get_n_files(path=path)
        return iteration
