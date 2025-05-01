from abc import abstractmethod
from typing import Generic, TypeVar

from artifact_core.base.artifact_dependencies import ArtifactResult

from artifact_experiment.base.tracking.logger import ArtifactLogger
from artifact_experiment.libs.tracking.filesystem.backend import (
    FilesystemBackend,
)

artifactResultT = TypeVar("artifactResultT", bound=ArtifactResult)


class FilesystemArtifactLogger(
    ArtifactLogger[artifactResultT, FilesystemBackend], Generic[artifactResultT]
):
    @abstractmethod
    def _log(self, path: str, artifact: artifactResultT): ...

    @classmethod
    @abstractmethod
    def _get_relative_path(cls, artifact_name: str) -> str: ...

    def _get_root_dir(self) -> str:
        return self._backend.native_client.run_dir
