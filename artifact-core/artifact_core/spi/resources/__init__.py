from artifact_core._base.core.resource_spec import (
    NO_RESOURCE_SPEC,
    NoResourceSpec,
    ResourceSpecProtocol,
)
from artifact_core._base.core.resources import ArtifactResources
from artifact_core._domains.classification.artifact import ClassificationArtifactResources
from artifact_core._domains.dataset_comparison.artifact import DatasetComparisonArtifactResources

__all__ = [
    "NO_RESOURCE_SPEC",
    "NoResourceSpec",
    "ResourceSpecProtocol",
    "ArtifactResources",
    "ClassificationArtifactResources",
    "DatasetComparisonArtifactResources",
]
