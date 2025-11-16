from artifact_experiment._base.primitives.file import File
from artifact_experiment._impl.backends.in_memory.loggers.base import InMemoryLogger


class InMemoryFileLogger(InMemoryLogger[File]):
    def _append(self, item_path: str, item: File):
        step = 1 + len(self._run.search_file_store(store_path=item_path))
        key = self._get_store_key(item_path=item_path, step=step)
        self._run.log_file(key=key, file=item)

    @classmethod
    def _get_relative_path(cls, item_name: str) -> str:
        return item_name
