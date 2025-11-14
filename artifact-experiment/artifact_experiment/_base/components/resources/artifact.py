from dataclasses import dataclass
from typing import Generic, TypeVar

from artifact_core.spi.resources import ArtifactResources

from artifact_experiment._base.components.resources.tracking import TrackingCallbackResources

ArtifactResourcesTCov = TypeVar("ArtifactResourcesTCov", bound=ArtifactResources, covariant=True)


@dataclass(frozen=True)
class ArtifactCallbackResources(TrackingCallbackResources, Generic[ArtifactResourcesTCov]):
    artifact_resources: ArtifactResourcesTCov
