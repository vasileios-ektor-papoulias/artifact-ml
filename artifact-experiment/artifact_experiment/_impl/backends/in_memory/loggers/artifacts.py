from abc import abstractmethod
from typing import TypeVar

from artifact_core.typing import ArtifactResult

from artifact_experiment._impl.backends.in_memory.loggers.base import InMemoryLogger

ArtifactResultT = TypeVar("ArtifactResultT", bound=ArtifactResult)


class InMemoryArtifactLogger(InMemoryLogger[ArtifactResultT]):
    @abstractmethod
    def _append(self, item_path: str, item: ArtifactResultT): ...

    @classmethod
    @abstractmethod
    def _get_relative_path(cls, item_name: str) -> str: ...
