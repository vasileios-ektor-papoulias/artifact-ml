from dataclasses import dataclass

from artifact_core._base.core.resources import ArtifactResources


@dataclass(frozen=True)
class DummyArtifactResources(ArtifactResources):
    valid: bool
    x: float

