from dataclasses import dataclass
from typing import Tuple

from artifact_core._base.core.hyperparams import ArtifactHyperparams
from artifact_core._domains.dataset_comparison.artifact import DatasetComparisonArtifact

from tests._domains.dataset_comparison.dummy.resource_spec import DummyResourceSpec
from tests._domains.dataset_comparison.dummy.resources import DummyDataset


@dataclass(frozen=True)
class DummyDatasetComparisonScoreHyperparams(ArtifactHyperparams):
    adjust_scale: bool


class DummyDatasetComparisonScore(
    DatasetComparisonArtifact[
        DummyDataset, DummyResourceSpec, DummyDatasetComparisonScoreHyperparams, float
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
        assert real_total != "invalid"
        real_synthetic = dataset_synthetic.x
        assert real_synthetic != "invalid"
        difference = real_total - real_synthetic
        if self._hyperparams.adjust_scale:
            difference = self._resource_spec.scale * difference
        result = abs(difference)
        return result
