from typing import TypeVar

from artifact_core.base.artifact_dependencies import (
    ArtifactResult,
    ResourceSpecProtocol,
)
from artifact_core.base.registry import ArtifactRegistry, ArtifactType
from artifact_core.core.dataset_comparison.artifact import (
    DatasetComparisonArtifactResources,
)

DatasetT = TypeVar("DatasetT")

ResourceSpecProtocolT = TypeVar("ResourceSpecProtocolT", bound=ResourceSpecProtocol)
ArtifactTypeT = TypeVar("ArtifactTypeT", bound=ArtifactType)
ArtifactResultT = TypeVar("ArtifactResultT", bound=ArtifactResult)


DatasetComparisonArtifactRegistry = ArtifactRegistry[
    DatasetComparisonArtifactResources[DatasetT],
    ResourceSpecProtocolT,
    ArtifactTypeT,
    ArtifactResultT,
]
