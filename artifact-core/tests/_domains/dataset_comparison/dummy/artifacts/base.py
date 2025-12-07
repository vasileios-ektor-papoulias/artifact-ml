from typing import TypeVar

from artifact_core._base.core.hyperparams import ArtifactHyperparams
from artifact_core._base.typing.artifact_result import ArtifactResult
from artifact_core._domains.dataset_comparison.artifact import DatasetComparisonArtifact

from tests._domains.dataset_comparison.dummy.resource_spec import DummyDatasetSpec
from tests._domains.dataset_comparison.dummy.resources import DummyDataset

ArtifactHyperparamsT = TypeVar("ArtifactHyperparamsT", bound=ArtifactHyperparams)
ArtifactResultT = TypeVar("ArtifactResultT", bound=ArtifactResult)

DummyDatasetComparisonArtifact = DatasetComparisonArtifact[
    DummyDataset, DummyDatasetSpec, ArtifactHyperparamsT, ArtifactResultT
]
