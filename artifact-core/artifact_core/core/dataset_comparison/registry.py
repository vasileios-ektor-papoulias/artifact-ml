from typing import TypeVar

from artifact_core.base.artifact_dependencies import (
    ArtifactResult,
    ResourceSpecProtocol,
)
from artifact_core.base.registry import ArtifactRegistry, ArtifactType
from artifact_core.core.dataset_comparison.artifact import (
    DatasetComparisonArtifactResources,
)

ArtifactTypeT = TypeVar("ArtifactTypeT", bound="ArtifactType")
DatasetT = TypeVar("DatasetT")
ArtifactResultT = TypeVar("ArtifactResultT", bound=ArtifactResult)
ResourceSpecProtocolT = TypeVar("ResourceSpecProtocolT", bound=ResourceSpecProtocol)


DatasetComparisonArtifactRegistry = ArtifactRegistry[
    ArtifactTypeT,
    DatasetComparisonArtifactResources[DatasetT],
    ArtifactResultT,
    ResourceSpecProtocolT,
]
