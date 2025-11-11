from abc import abstractmethod
from typing import TypeVar

from artifact_core._base.artifact_dependencies import ArtifactResult

from artifact_experiment.libs.tracking.clear_ml.loggers.base import ClearMLLogger

ArtifactResultT = TypeVar("ArtifactResultT", bound=ArtifactResult)


class ClearMLArtifactLogger(ClearMLLogger[ArtifactResultT]):
    _root_dir = "artifacts"

    @abstractmethod
    def _append(self, item_path: str, item: ArtifactResultT): ...

    @classmethod
    @abstractmethod
    def _get_relative_path(cls, item_name: str) -> str: ...

    def _get_root_dir(self) -> str:
        return self._root_dir
