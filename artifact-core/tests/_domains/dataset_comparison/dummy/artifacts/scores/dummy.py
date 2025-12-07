from dataclasses import dataclass
from typing import Tuple

from artifact_core._base.core.hyperparams import ArtifactHyperparams

from tests._domains.dataset_comparison.dummy.artifacts.base import DummyDatasetComparisonArtifact
from tests._domains.dataset_comparison.dummy.registries.scores import (
    DummyDatasetComparisonScoreRegistry,
)
from tests._domains.dataset_comparison.dummy.resource_spec import DummyDatasetSpec
from tests._domains.dataset_comparison.dummy.resources import DummyDataset
from tests._domains.dataset_comparison.dummy.types.scores import DummyDatasetComparisonScoreType


@DummyDatasetComparisonScoreRegistry.register_artifact_hyperparams(
    artifact_type=DummyDatasetComparisonScoreType.DUMMY_SCORE
)
@dataclass(frozen=True)
class DummyDatasetComparisonScoreHyperparams(ArtifactHyperparams):
    adjust_scale: bool


@DummyDatasetComparisonScoreRegistry.register_artifact(
    artifact_type=DummyDatasetComparisonScoreType.DUMMY_SCORE
)
class DummyDatasetComparisonScore(
    DummyDatasetComparisonArtifact[DummyDatasetComparisonScoreHyperparams, float]
):
    def __init__(
        self, resource_spec: DummyDatasetSpec, hyperparams: DummyDatasetComparisonScoreHyperparams
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
        assert real_total != "invalid"
        real_synthetic = dataset_synthetic.x
        assert real_synthetic != "invalid"
        difference = real_total - real_synthetic
        if self._hyperparams.adjust_scale:
            difference = self._resource_spec.scale * difference
        result = abs(difference)
        return result
