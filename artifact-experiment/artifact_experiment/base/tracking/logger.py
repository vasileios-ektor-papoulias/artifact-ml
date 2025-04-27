import os
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from artifact_core.base.artifact_dependencies import ArtifactResult

from artifact_experiment.base.tracking.backend import TrackingBackend

artifactResultT = TypeVar("artifactResultT", bound=ArtifactResult)
trackingBackendT = TypeVar("trackingBackendT", bound=TrackingBackend)


class ArtifactLogger(ABC, Generic[artifactResultT, trackingBackendT]):
    _default_root_dir = "artifact_ml"

    def __init__(self, backend: trackingBackendT):
        self._backend = backend

    @abstractmethod
    def _log(self, path: str, artifact: artifactResultT):
        pass

    @classmethod
    @abstractmethod
    def _get_relative_path(cls, artifact_name: str) -> str:
        pass

    @classmethod
    def _get_artifact_path(cls, artifact_name: str) -> str:
        root_dir = cls._get_root_dir()
        relative_path = cls._get_relative_path(artifact_name=artifact_name)
        artifact_path = os.path.join(root_dir, relative_path)
        return artifact_path

    @classmethod
    def _get_root_dir(cls) -> str:
        return cls._default_root_dir

    def log(self, name: str, artifact: artifactResultT):
        path = self._get_artifact_path(artifact_name=name)
        self._log(path=path, artifact=artifact)
