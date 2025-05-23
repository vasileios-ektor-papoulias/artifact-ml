from abc import abstractmethod
from typing import Generic, TypeVar

from artifact_core.base.artifact_dependencies import ArtifactResult

from artifact_experiment.base.tracking.logger import ArtifactLogger
from artifact_experiment.libs.tracking.clear_ml.adapter import (
    ClearMLRunAdapter,
)

artifactResultT = TypeVar("artifactResultT", bound=ArtifactResult)


class ClearMLArtifactLogger(
    ArtifactLogger[artifactResultT, ClearMLRunAdapter], Generic[artifactResultT]
):
    _root_dir = "artifact_ml"

    @abstractmethod
    def _append(self, artifact_path: str, artifact: artifactResultT): ...

    @classmethod
    @abstractmethod
    def _get_relative_path(cls, artifact_name: str) -> str: ...

    def _get_root_dir(self) -> str:
        return self._root_dir
