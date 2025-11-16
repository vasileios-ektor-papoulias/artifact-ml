from dataclasses import dataclass

from artifact_core._domains.dataset_comparison.artifact import (
    DatasetComparisonArtifactResources,
    ResourceSpecProtocol,
)


@dataclass
class DummyResourceSpec(ResourceSpecProtocol):
    scale: float


@dataclass
class DummyDataset:
    x: float


DummyDatasetComparisonArtifactResources = DatasetComparisonArtifactResources[DummyDataset]
