from abc import abstractmethod
from typing import Generic, TypeVar

from artifact_core.base.artifact_dependencies import ArtifactResult

from artifact_experiment.base.tracking.logger import ArtifactLogger
from artifact_experiment.libs.tracking.neptune.adapter import (
    NeptuneRunAdapter,
)

ArtifactResultT = TypeVar("ArtifactResultT", bound=ArtifactResult)


class NeptuneArtifactLogger(
    ArtifactLogger[ArtifactResultT, NeptuneRunAdapter], Generic[ArtifactResultT]
):
    _root_dir = "artifacts"

    def _append(self, artifact_path: str, artifact: ArtifactResultT):
        self._run.log(path=artifact_path, artifact=artifact)

    @classmethod
    @abstractmethod
    def _get_relative_path(cls, artifact_name: str) -> str: ...

    def _get_root_dir(self) -> str:
        return self._root_dir
