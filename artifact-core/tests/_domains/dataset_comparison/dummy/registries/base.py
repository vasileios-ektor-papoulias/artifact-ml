from typing import TypeVar

from artifact_core._base.typing.artifact_result import ArtifactResult
from artifact_core._domains.dataset_comparison.registry import (
    ArtifactType,
    DatasetComparisonArtifactRegistry,
)

from tests._domains.dataset_comparison.dummy.resource_spec import DummyDatasetSpec
from tests._domains.dataset_comparison.dummy.resources import DummyDataset

ArtifactTypeT = TypeVar("ArtifactTypeT", bound=ArtifactType)
ArtifactResultT = TypeVar("ArtifactResultT", bound=ArtifactResult)

DummyDatasetComparisonRegistry = DatasetComparisonArtifactRegistry[
    DummyDataset, DummyDatasetSpec, ArtifactTypeT, ArtifactResultT
]
