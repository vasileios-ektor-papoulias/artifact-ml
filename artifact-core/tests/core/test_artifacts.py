import pytest

from tests.core.dummy.artifact_dependencies import (
    DummyDataset,
    DummyDatasetComparisonArtifactResources,
    DummyDataSpec,
)
from tests.core.dummy.artifacts import (
    DummyDatasetComparisonScore,
    DummyDatasetComparisonScoreHyperparams,
)


@pytest.mark.parametrize(
    "hyperparams, data_spec, artifact_resources, expected, expect_raise_invalid_resource",
    [
        (
            DummyDatasetComparisonScoreHyperparams(adjust_scale=True),
            DummyDataSpec(scale=1),
            DummyDatasetComparisonArtifactResources(
                dataset_real=DummyDataset(x=1.0), dataset_synthetic=DummyDataset(x=1.0)
            ),
            0,
            False,
        ),
        (
            DummyDatasetComparisonScoreHyperparams(adjust_scale=True),
            DummyDataSpec(scale=1),
            DummyDatasetComparisonArtifactResources(
                dataset_real=DummyDataset(x=1), dataset_synthetic=DummyDataset(x=1.0)
            ),
            0,
            True,
        ),
        (
            DummyDatasetComparisonScoreHyperparams(adjust_scale=True),
            DummyDataSpec(scale=1),
            DummyDatasetComparisonArtifactResources(
                dataset_real=DummyDataset(x=1.0), dataset_synthetic=DummyDataset(x=1)
            ),
            0,
            True,
        ),
        (
            DummyDatasetComparisonScoreHyperparams(adjust_scale=True),
            DummyDataSpec(scale=1),
            DummyDatasetComparisonArtifactResources(
                dataset_real=DummyDataset(x="invalid"), dataset_synthetic=DummyDataset(x="invalid")
            ),
            0,
            True,
        ),
        (
            DummyDatasetComparisonScoreHyperparams(adjust_scale=True),
            DummyDataSpec(scale=2),
            DummyDatasetComparisonArtifactResources(
                dataset_real=DummyDataset(x=1), dataset_synthetic=DummyDataset(x=3)
            ),
            4,
            True,
        ),
    ],
)
def test_compute(
    hyperparams: DummyDatasetComparisonScoreHyperparams,
    data_spec: DummyDataSpec,
    artifact_resources: DummyDatasetComparisonArtifactResources,
    expected: float,
    expect_raise_invalid_resource: bool,
):
    artifact = DummyDatasetComparisonScore(data_spec=data_spec, hyperparams=hyperparams)
    if expect_raise_invalid_resource:
        with pytest.raises(ValueError, match="Invalid Data: expected float, got"):
            artifact.compute(resources=artifact_resources)
    else:
        result = artifact.compute(resources=artifact_resources)
        assert result == expected
