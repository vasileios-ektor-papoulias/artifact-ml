from abc import ABC, abstractmethod
from typing import Generic, Optional, TypeVar

from artifact_core.base.artifact_dependencies import (
    ArtifactHyperparams,
    ArtifactResources,
    ArtifactResult,
    DataSpecProtocol,
)

dataSpecProtocolT = TypeVar("dataSpecProtocolT", bound=DataSpecProtocol)
artifactHyperparamsT = TypeVar("artifactHyperparamsT", bound="ArtifactHyperparams")
artifactResourcesT = TypeVar("artifactResourcesT", bound="ArtifactResources")
artifactResultT = TypeVar("artifactResultT", bound=ArtifactResult)


class Artifact(
    ABC,
    Generic[
        artifactResourcesT,
        artifactResultT,
        artifactHyperparamsT,
        dataSpecProtocolT,
    ],
):
    def __init__(
        self,
        data_spec: Optional[dataSpecProtocolT] = None,
        hyperparams: Optional[artifactHyperparamsT] = None,
    ):
        self._data_spec = data_spec
        self._hyperparams = hyperparams

    def __call__(self, resources: artifactResourcesT) -> artifactResultT:
        return self.compute(resources=resources)

    @abstractmethod
    def _compute(self, resources: artifactResourcesT) -> artifactResultT:
        pass

    @abstractmethod
    def _validate(self, resources: artifactResourcesT) -> artifactResourcesT:
        pass

    def compute(self, resources: artifactResourcesT) -> artifactResultT:
        resources = self._validate(resources=resources)
        result = self._compute(resources=resources)
        return result
