from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from artifact_core.base.artifact import Artifact
from artifact_core.base.artifact_dependencies import (
    ArtifactResources,
    ArtifactResult,
    ResourceSpecProtocol,
)
from artifact_experiment.base.callbacks.base import (
    CallbackResources,
)
from artifact_experiment.base.callbacks.tracking import (
    TrackingCallback,
)

resourceSpecProtocolT = TypeVar("resourceSpecProtocolT", bound=ResourceSpecProtocol)
artifactResourcesT = TypeVar("artifactResourcesT", bound=ArtifactResources)
artifactResultT = TypeVar("artifactResultT", bound=ArtifactResult)


@dataclass
class ArtifactCallbackResources(CallbackResources, Generic[artifactResourcesT]):
    artifact_resources: artifactResourcesT


class ArtifactCallback(
    TrackingCallback[ArtifactCallbackResources[artifactResourcesT], artifactResultT],
    Generic[artifactResourcesT, artifactResultT, resourceSpecProtocolT],
):
    def __init__(
        self,
        key: str,
        artifact: Artifact[artifactResourcesT, artifactResultT, Any, resourceSpecProtocolT],
    ):
        super().__init__(key=key)
        self._artifact = artifact

    def _compute(self, resources: ArtifactCallbackResources[artifactResourcesT]) -> artifactResultT:
        result = self._artifact.compute(resources=resources.artifact_resources)
        return result
