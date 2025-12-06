from typing import TypeVar

from artifact_core._base.core.resource_spec import ResourceSpecProtocol
from artifact_core._base.orchestration.registry import ArtifactRegistry
from artifact_core._base.primitives.artifact_type import ArtifactType
from artifact_core._base.typing.artifact_result import ArtifactResult
from artifact_core._domains.dataset_comparison.artifact import DatasetComparisonArtifactResources

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
