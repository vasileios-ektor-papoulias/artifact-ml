from dataclasses import dataclass
from typing import Literal, Union

from artifact_core._domains.dataset_comparison.artifact import DatasetComparisonArtifactResources


@dataclass
class DummyDataset:
    x: Union[float, Literal["invalid"]]


DummyDatasetComparisonArtifactResources = DatasetComparisonArtifactResources[DummyDataset]
