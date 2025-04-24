from dataclasses import dataclass

from artifact_core.base.artifact_dependencies import (
    ArtifactResources,
    ResourceSpecProtocol,
)


@dataclass
class DummyResourceSpec(ResourceSpecProtocol):
    scale: float


@dataclass(frozen=True)
class DummyArtifactResources(ArtifactResources):
    valid: bool
    x: float
