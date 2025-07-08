from abc import abstractmethod
from typing import Generic, TypeVar

from artifact_core.base.artifact_dependencies import ArtifactResult

from artifact_experiment.base.tracking.logger import ArtifactLogger
from artifact_experiment.libs.tracking.in_memory.adapter import (
    InMemoryRunAdapter,
)

ArtifactResultT = TypeVar("ArtifactResultT", bound=ArtifactResult)


class InMemoryArtifactLogger(
    ArtifactLogger[ArtifactResultT, InMemoryRunAdapter], Generic[ArtifactResultT]
):
    @abstractmethod
    def _append(self, artifact_path: str, artifact: ArtifactResultT): ...

    @classmethod
    @abstractmethod
    def _get_relative_path(cls, artifact_name: str) -> str: ...

    def _get_root_dir(self) -> str:
        return f"{self._run.experiment_id}/{self._run.run_id}"

    @staticmethod
    def _get_store_key(artifact_path: str, step: int) -> str:
        return f"{artifact_path}/{step}".replace("\\", "/")
