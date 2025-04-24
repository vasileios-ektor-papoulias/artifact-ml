from typing import TypeVar

from artifact_core.base.artifact_dependencies import (
    ArtifactResult,
    ResourceSpecProtocol,
)
from artifact_core.base.registry import ArtifactRegistry, ArtifactType
from artifact_core.core.dataset_comparison.artifact import (
    DatasetComparisonArtifactResources,
)

artifactTypeT = TypeVar("artifactTypeT", bound="ArtifactType")
datasetT = TypeVar("datasetT")
artifactResultT = TypeVar("artifactResultT", bound=ArtifactResult)
resourceSpecProtocolT = TypeVar("resourceSpecProtocolT", bound=ResourceSpecProtocol)


DatasetComparisonArtifactRegistry = ArtifactRegistry[
    artifactTypeT,
    DatasetComparisonArtifactResources[datasetT],
    artifactResultT,
    resourceSpecProtocolT,
]
