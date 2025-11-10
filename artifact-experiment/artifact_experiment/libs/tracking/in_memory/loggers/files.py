from artifact_experiment.base.entities.file import File
from artifact_experiment.libs.tracking.in_memory.loggers.base import InMemoryLogger


class InMemoryFileLogger(InMemoryLogger[File]):
    def _append(self, item_path: str, item: File):
        self._run.upload(path_source=item.path_source, dir_target=item_path)

    @classmethod
    def _get_relative_path(cls, item_name: str) -> str:
        return item_name
