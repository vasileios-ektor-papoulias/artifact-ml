from dataclasses import dataclass
from typing import Tuple, TypeVar

from artifact_core.base.artifact_dependencies import (
    ArtifactHyperparams,
    ArtifactResult,
)
from artifact_core.core.dataset_comparison.artifact import DatasetComparisonArtifact

from tests.core.dataset_comparison.dummy.artifact_dependencies import (
    DummyDataset,
    DummyResourceSpec,
)

artifactHyperparamsT = TypeVar("artifactHyperparamsT", bound=ArtifactHyperparams)
artifactResultT = TypeVar("artifactResultT", bound=ArtifactResult)


@dataclass(frozen=True)
class DummyDatasetComparisonScoreHyperparams(ArtifactHyperparams):
    adjust_scale: bool


class DummyDatasetComparisonScore(
    DatasetComparisonArtifact[
        DummyDataset, float, DummyDatasetComparisonScoreHyperparams, DummyResourceSpec
    ]
):
    def __init__(
        self, resource_spec: DummyResourceSpec, hyperparams: DummyDatasetComparisonScoreHyperparams
    ):
        self._resource_spec = resource_spec
        self._hyperparams = hyperparams

    def _validate_datasets(
        self, dataset_real: DummyDataset, dataset_synthetic: DummyDataset
    ) -> Tuple[DummyDataset, DummyDataset]:
        if not isinstance(dataset_real.x, float):
            raise ValueError(f"Invalid Data: expected float, got {type(dataset_real.x)}")
        if not isinstance(dataset_synthetic.x, float):
            raise ValueError(f"Invalid Data: expected float, got {type(dataset_synthetic.x)}")
        return (dataset_real, dataset_synthetic)

    def _compare_datasets(
        self, dataset_real: DummyDataset, dataset_synthetic: DummyDataset
    ) -> float:
        real_total = dataset_real.x
        real_synthetic = dataset_synthetic.x
        difference = real_total - real_synthetic
        if self._hyperparams.adjust_scale:
            difference = self._resource_spec.scale * difference
        result = abs(difference)
        return result
