from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from artifact_core.base.artifact_dependencies import ArtifactResult

from artifact_experiment.base.tracking.backend import TrackingBackend

artifactResultT = TypeVar("artifactResultT", bound=ArtifactResult)
trackingBackendT = TypeVar("trackingBackendT", bound=TrackingBackend)


class ArtifactLogger(ABC, Generic[artifactResultT, trackingBackendT]):
    def __init__(self, backend: trackingBackendT):
        self._backend = backend

    @staticmethod
    @abstractmethod
    def _get_artifact_path(artifact_name: str) -> str:
        pass

    @abstractmethod
    def _log(self, path: str, artifact: artifactResultT):
        pass

    def log(self, name: str, artifact: artifactResultT):
        path = self._get_artifact_path(artifact_name=name)
        self._log(path=path, artifact=artifact)
