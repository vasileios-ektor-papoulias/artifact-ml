from dataclasses import dataclass

from artifact_core._base.artifact_dependencies import (
    ArtifactResources,
    ResourceSpecProtocol,
)


@dataclass
class DummyResourceSpec(ResourceSpecProtocol):
    pass


@dataclass(frozen=True)
class DummyArtifactResources(ArtifactResources):
    pass
