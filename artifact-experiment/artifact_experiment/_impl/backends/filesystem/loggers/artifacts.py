import os
from abc import abstractmethod
from typing import TypeVar

from artifact_core.typing import ArtifactResult

from artifact_experiment._impl.backends.filesystem.loggers.base import FilesystemLogger

ArtifactResultT = TypeVar("ArtifactResultT", bound=ArtifactResult)


class FilesystemArtifactLogger(FilesystemLogger[ArtifactResultT]):
    @abstractmethod
    def _append(self, item_path: str, item: ArtifactResultT): ...

    @classmethod
    @abstractmethod
    def _get_relative_path(cls, item_name: str) -> str: ...

    def _get_root_dir(self) -> str:
        return os.path.join(self._run.run_dir, "artifacts")
