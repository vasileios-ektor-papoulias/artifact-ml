from dataclasses import dataclass

from artifact_core.core.dataset_comparison.artifact import (
    DatasetComparisonArtifactResources,
    DataSpecProtocol,
)


@dataclass
class DummyDataSpec(DataSpecProtocol):
    scale: float


@dataclass
class DummyDataset:
    x: float


DummyDatasetComparisonArtifactResources = DatasetComparisonArtifactResources[DummyDataset]
