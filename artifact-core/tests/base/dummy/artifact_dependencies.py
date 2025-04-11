from dataclasses import dataclass

from artifact_core.base.artifact_dependencies import (
    ArtifactResources,
    DataSpecProtocol,
)


@dataclass
class DummyDataSpec(DataSpecProtocol):
    scale: float


@dataclass(frozen=True)
class DummyArtifactResources(ArtifactResources):
    valid: bool
    x: float
