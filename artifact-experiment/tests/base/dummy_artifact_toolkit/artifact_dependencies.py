from dataclasses import dataclass

from artifact_core.base.artifact_dependencies import (
    ArtifactResources,
    ResourceSpecProtocol,
)


@dataclass
class DummyResourceSpec(ResourceSpecProtocol):
    pass


@dataclass(frozen=True)
class DummyArtifactResources(ArtifactResources):
    pass
