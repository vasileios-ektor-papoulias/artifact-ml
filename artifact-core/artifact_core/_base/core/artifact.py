from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from artifact_core._base.core.hyperparams import ArtifactHyperparams
from artifact_core._base.core.resource_spec import ResourceSpecProtocol
from artifact_core._base.core.resources import ArtifactResources
from artifact_core._base.typing.artifact_result import ArtifactResult

ArtifactResourcesT = TypeVar("ArtifactResourcesT", bound=ArtifactResources)
ResourceSpecProtocolT = TypeVar("ResourceSpecProtocolT", bound=ResourceSpecProtocol)
ArtifactHyperparamsT = TypeVar("ArtifactHyperparamsT", bound=ArtifactHyperparams)
ArtifactResultT = TypeVar("ArtifactResultT", bound=ArtifactResult)


class Artifact(
    ABC, Generic[ArtifactResourcesT, ResourceSpecProtocolT, ArtifactHyperparamsT, ArtifactResultT]
):
    def __init__(self, resource_spec: ResourceSpecProtocolT, hyperparams: ArtifactHyperparamsT):
        self._resource_spec = resource_spec
        self._hyperparams = hyperparams

    @property
    def hyperparams(self) -> ArtifactHyperparamsT:
        return self._hyperparams

    @property
    def resource_spec(self) -> ResourceSpecProtocolT:
        return self._resource_spec

    def __call__(self, resources: ArtifactResourcesT) -> ArtifactResultT:
        return self.compute(resources=resources)

    @abstractmethod
    def _compute(self, resources: ArtifactResourcesT) -> ArtifactResultT:
        pass

    @abstractmethod
    def _validate(self, resources: ArtifactResourcesT) -> ArtifactResourcesT:
        pass

    def compute(self, resources: ArtifactResourcesT) -> ArtifactResultT:
        resources = self._validate(resources=resources)
        result = self._compute(resources=resources)
        return result
