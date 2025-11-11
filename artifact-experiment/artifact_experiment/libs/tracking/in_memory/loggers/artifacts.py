from abc import abstractmethod
from typing import TypeVar

from artifact_core._base.artifact_dependencies import ArtifactResult

from artifact_experiment.libs.tracking.in_memory.loggers.base import InMemoryLogger

ArtifactResultT = TypeVar("ArtifactResultT", bound=ArtifactResult)


class InMemoryArtifactLogger(InMemoryLogger[ArtifactResultT]):
    @abstractmethod
    def _append(self, item_path: str, item: ArtifactResultT): ...

    @classmethod
    @abstractmethod
    def _get_relative_path(cls, item_name: str) -> str: ...

    @staticmethod
    def _get_store_key(item_path: str, step: int) -> str:
        return f"{item_path}/{step}"
