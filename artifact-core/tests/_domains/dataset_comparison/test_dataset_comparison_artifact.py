import pytest
from artifact_core._domains.dataset_comparison.artifact import DatasetComparisonArtifactResources

from tests._domains.dataset_comparison.dummy.artifacts.scores.dummy import (
    DummyDatasetComparisonScore,
    DummyDatasetComparisonScoreHyperparams,
)
from tests._domains.dataset_comparison.dummy.resource_spec import DummyDatasetSpec
from tests._domains.dataset_comparison.dummy.resources import DummyDataset

DummyDatasetComparisonArtifactResources = DatasetComparisonArtifactResources[DummyDataset]


@pytest.mark.unit
@pytest.mark.parametrize(
    "hyperparams, resource_spec, artifact_resources, expected",
    [
        (
            DummyDatasetComparisonScoreHyperparams(adjust_scale=True),
            DummyDatasetSpec(scale=1),
            DummyDatasetComparisonArtifactResources(
                dataset_real=DummyDataset(x=1.0), dataset_synthetic=DummyDataset(x=1.0)
            ),
            0,
        ),
        (
            DummyDatasetComparisonScoreHyperparams(adjust_scale=True),
            DummyDatasetSpec(scale=2),
            DummyDatasetComparisonArtifactResources(
                dataset_real=DummyDataset(x=1.0), dataset_synthetic=DummyDataset(x=1.0)
            ),
            0,
        ),
        (
            DummyDatasetComparisonScoreHyperparams(adjust_scale=True),
            DummyDatasetSpec(scale=2),
            DummyDatasetComparisonArtifactResources(
                dataset_real=DummyDataset(x=3.0), dataset_synthetic=DummyDataset(x=1.0)
            ),
            4,
        ),
        (
            DummyDatasetComparisonScoreHyperparams(adjust_scale=False),
            DummyDatasetSpec(scale=2),
            DummyDatasetComparisonArtifactResources(
                dataset_real=DummyDataset(x=5.0), dataset_synthetic=DummyDataset(x=2.0)
            ),
            3,
        ),
    ],
)
def test_compute(
    hyperparams: DummyDatasetComparisonScoreHyperparams,
    resource_spec: DummyDatasetSpec,
    artifact_resources: DummyDatasetComparisonArtifactResources,
    expected: float,
):
    artifact = DummyDatasetComparisonScore(resource_spec=resource_spec, hyperparams=hyperparams)
    result = artifact.compute(resources=artifact_resources)
    assert result == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    "hyperparams, resource_spec, artifact_resources",
    [
        (
            DummyDatasetComparisonScoreHyperparams(adjust_scale=True),
            DummyDatasetSpec(scale=1),
            DummyDatasetComparisonArtifactResources(
                dataset_real=DummyDataset(x=1), dataset_synthetic=DummyDataset(x=1.0)
            ),
        ),
        (
            DummyDatasetComparisonScoreHyperparams(adjust_scale=True),
            DummyDatasetSpec(scale=1),
            DummyDatasetComparisonArtifactResources(
                dataset_real=DummyDataset(x=1.0), dataset_synthetic=DummyDataset(x=1)
            ),
        ),
        (
            DummyDatasetComparisonScoreHyperparams(adjust_scale=True),
            DummyDatasetSpec(scale=1),
            DummyDatasetComparisonArtifactResources(
                dataset_real=DummyDataset(x="invalid"), dataset_synthetic=DummyDataset(x="invalid")
            ),
        ),
        (
            DummyDatasetComparisonScoreHyperparams(adjust_scale=True),
            DummyDatasetSpec(scale=2),
            DummyDatasetComparisonArtifactResources(
                dataset_real=DummyDataset(x="invalid"), dataset_synthetic=DummyDataset(x=1.0)
            ),
        ),
    ],
)
def test_compute_invalid_data(
    hyperparams: DummyDatasetComparisonScoreHyperparams,
    resource_spec: DummyDatasetSpec,
    artifact_resources: DummyDatasetComparisonArtifactResources,
):
    artifact = DummyDatasetComparisonScore(resource_spec=resource_spec, hyperparams=hyperparams)
    with pytest.raises(ValueError, match="Invalid Data: expected float, got"):
        artifact.compute(resources=artifact_resources)
