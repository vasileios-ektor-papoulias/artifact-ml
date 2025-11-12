from dataclasses import dataclass

from artifact_core._base.primitives import (
    ArtifactResources,
    ResourceSpecProtocol,
)


@dataclass
class DummyResourceSpec(ResourceSpecProtocol):
    pass


@dataclass(frozen=True)
class DummyArtifactResources(ArtifactResources):
    pass
