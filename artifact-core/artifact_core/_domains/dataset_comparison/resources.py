from dataclasses import dataclass
from typing import Generic, TypeVar

from artifact_core._base.core.resources import ArtifactResources

DatasetT = TypeVar("DatasetT")


@dataclass(frozen=True)
class DatasetComparisonArtifactResources(ArtifactResources, Generic[DatasetT]):
    dataset_real: DatasetT
    dataset_synthetic: DatasetT
