from artifact_experiment.base.entities.file import File
from artifact_experiment.libs.tracking.neptune.loggers.base import NeptuneLogger


class NeptuneFileLogger(NeptuneLogger[File]):
    _root_dir = "files"

    def _append(self, item_path: str, item: File):
        self._run.upload(path_source=item.path_source, dir_target=item_path)

    @classmethod
    def _get_relative_path(cls, item_name: str) -> str:
        return item_name

    def _get_root_dir(self) -> str:
        return self._root_dir
