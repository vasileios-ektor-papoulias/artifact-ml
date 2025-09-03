from typing import TypeVar

from artifact_core.base.artifact_dependencies import (
    ArtifactResult,
    ResourceSpecProtocol,
)
from artifact_core.base.registry import ArtifactRegistry, ArtifactType
from artifact_core.core.classification.artifact import (
    ClassificationArtifactResources,
)

ArtifactTypeT = TypeVar("ArtifactTypeT", bound="ArtifactType")
LabelsT = TypeVar("LabelsT")
ArtifactResultT = TypeVar("ArtifactResultT", bound=ArtifactResult)
ResourceSpecProtocolT = TypeVar("ResourceSpecProtocolT", bound=ResourceSpecProtocol)


ClassificationArtifactRegistry = ArtifactRegistry[
    ArtifactTypeT,
    ClassificationArtifactResources[LabelsT],
    ArtifactResultT,
    ResourceSpecProtocolT,
]
