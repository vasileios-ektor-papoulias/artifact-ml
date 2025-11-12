from artifact_core._base.contracts.resource_spec import (
    NO_RESOURCE_SPEC,
    NoResourceSpec,
    ResourceSpecProtocol,
)
from artifact_core._base.contracts.resources import ArtifactResources
from artifact_core._tasks.classification.artifact import ClassificationArtifactResources
from artifact_core._tasks.dataset_comparison.artifact import DatasetComparisonArtifactResources

__all__ = [
    "NO_RESOURCE_SPEC",
    "NoResourceSpec",
    "ResourceSpecProtocol",
    "ArtifactResources",
    "ClassificationArtifactResources",
    "DatasetComparisonArtifactResources",
]
