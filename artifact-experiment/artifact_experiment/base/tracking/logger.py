import os
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from artifact_core.base.artifact_dependencies import ArtifactResult

from artifact_experiment.base.tracking.adapter import RunAdapter

artifactResultT = TypeVar("artifactResultT", bound=ArtifactResult)
runAdapterT = TypeVar("runAdapterT", bound=RunAdapter)


class ArtifactLogger(ABC, Generic[artifactResultT, runAdapterT]):
    def __init__(self, run: runAdapterT):
        self._run = run

    @abstractmethod
    def _log(self, path: str, artifact: artifactResultT):
        pass

    @classmethod
    @abstractmethod
    def _get_relative_path(cls, artifact_name: str) -> str:
        pass

    @abstractmethod
    def _get_root_dir(self) -> str: ...

    @classmethod
    def _get_artifact_path(cls, root_dir: str, artifact_name: str) -> str:
        relative_path = cls._get_relative_path(artifact_name=artifact_name)
        artifact_path = os.path.join(root_dir, relative_path)
        return artifact_path

    def log(self, name: str, artifact: artifactResultT):
        root_dir = self._get_root_dir()
        path = self._get_artifact_path(root_dir=root_dir, artifact_name=name)
        self._log(path=path, artifact=artifact)
