import os
from abc import abstractmethod
from typing import Generic, TypeVar

from artifact_core.base.artifact_dependencies import ArtifactResult

from artifact_experiment.base.tracking.logger import ArtifactLogger
from artifact_experiment.libs.tracking.filesystem.adapter import (
    FilesystemRunAdapter,
)

artifactResultT = TypeVar("artifactResultT", bound=ArtifactResult)


class FilesystemArtifactLogger(
    ArtifactLogger[artifactResultT, FilesystemRunAdapter], Generic[artifactResultT]
):
    @abstractmethod
    def _append(self, artifact_path: str, artifact: artifactResultT): ...

    @classmethod
    @abstractmethod
    def _get_relative_path(cls, artifact_name: str) -> str: ...

    def _get_root_dir(self) -> str:
        return os.path.join(self._run.run_dir, "artifacts")
