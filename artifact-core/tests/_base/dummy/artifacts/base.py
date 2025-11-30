from typing import Optional, TypeVar

from artifact_core._base.core.artifact import Artifact
from artifact_core._base.core.hyperparams import ArtifactHyperparams
from artifact_core._base.typing.artifact_result import ArtifactResult

from tests._base.dummy.resource_spec import DummyResourceSpec
from tests._base.dummy.resources import DummyArtifactResources

ArtifactHyperparamsT = TypeVar("ArtifactHyperparamsT", bound=ArtifactHyperparams)
ArtifactResultT = TypeVar("ArtifactResultT", bound=ArtifactResult)


class DummyArtifact(
    Artifact[DummyArtifactResources, DummyResourceSpec, ArtifactHyperparamsT, ArtifactResultT]
):
    def __init__(
        self, resource_spec: DummyResourceSpec, hyperparams: Optional[ArtifactHyperparamsT] = None
    ):
        self._resource_spec = resource_spec
        self._hyperparams = hyperparams

    def _validate(self, resources: DummyArtifactResources) -> DummyArtifactResources:
        if not resources.valid:
            raise ValueError("Invalid Resources")
        return resources
