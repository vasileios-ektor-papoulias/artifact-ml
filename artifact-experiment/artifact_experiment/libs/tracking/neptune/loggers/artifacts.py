from abc import abstractmethod
from typing import TypeVar

from artifact_core._base.primitives import ArtifactResult

from artifact_experiment.libs.tracking.neptune.loggers.base import NeptuneLogger

ArtifactResultT = TypeVar("ArtifactResultT", bound=ArtifactResult)


class NeptuneArtifactLogger(NeptuneLogger[ArtifactResultT]):
    _root_dir = "artifacts"

    def _append(self, item_path: str, item: ArtifactResultT):
        self._run.log(artifact_path=item_path, artifact=item)

    @classmethod
    @abstractmethod
    def _get_relative_path(cls, item_name: str) -> str: ...

    def _get_root_dir(self) -> str:
        return self._root_dir
