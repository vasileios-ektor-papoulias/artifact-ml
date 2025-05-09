from abc import abstractmethod
from typing import Generic, TypeVar

from artifact_core.base.artifact_dependencies import ArtifactResult

from artifact_experiment.base.tracking.logger import ArtifactLogger
from artifact_experiment.libs.tracking.mlflow.adapter import MlflowRunAdapter

artifactResultT = TypeVar("artifactResultT", bound=ArtifactResult)


class MlflowArtifactLogger(
    ArtifactLogger[artifactResultT, MlflowRunAdapter], Generic[artifactResultT]
):
    _root_dir = "artifact_ml"

    @abstractmethod
    def _log(self, path: str, artifact: artifactResultT): ...

    @classmethod
    @abstractmethod
    def _get_relative_path(cls, artifact_name: str) -> str: ...

    @staticmethod
    def _get_next_step_count(run: MlflowRunAdapter, path: str) -> int:
        history = run.get_ls_score_history(backend_path=path)
        next_step = len(history) + 1
        return next_step

    def _get_root_dir(self) -> str:
        return self._root_dir
