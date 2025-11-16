from typing import Any, Dict

from artifact_experiment._base.primitives.file import File
from artifact_experiment._impl.backends.neptune.adapter import NeptuneRunAdapter
from artifact_experiment._impl.backends.neptune.loggers.base import NeptuneLogger
from artifact_experiment._utils.filesystem.basename_extractor import BasenameExtractor
from artifact_experiment._utils.filesystem.incremental_paths import IncrementalPathFormatter


class NeptuneFileLogger(NeptuneLogger[File]):
    _root_dir = "files"

    def _append(self, item_path: str, item: File):
        namespace_data = self._get_namespace_data(run=self._run, path=item_path)
        step = self._get_next_step_from_namespace_data(namespace_data=namespace_data)
        item_path = self._qualify_item_path_with_step(item_path=item_path, item=item, step=step)
        self._run.log_file(backend_dir=item_path, file=item)

    @staticmethod
    def _get_namespace_data(run: NeptuneRunAdapter, path: str) -> Dict[str, Any]:
        namespace_data = run.get_namespace_data(backend_path=path)
        return namespace_data

    @staticmethod
    def _get_next_step_from_namespace_data(namespace_data: Dict[str, Any]) -> int:
        return 1 + len(namespace_data)

    @staticmethod
    def _qualify_item_path_with_step(item_path: str, item: File, step: int) -> str:
        filename = BasenameExtractor.extract(path=item.path_source)
        qualified_path = IncrementalPathFormatter.format(dir_path=item_path, next_idx=step)
        return f"{qualified_path}/{filename}"

    @classmethod
    def _get_relative_path(cls, item_name: str) -> str:
        return item_name

    def _get_root_dir(self) -> str:
        return self._root_dir
