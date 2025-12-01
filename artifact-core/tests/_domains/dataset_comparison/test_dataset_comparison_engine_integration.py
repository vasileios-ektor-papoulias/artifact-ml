import pytest

from tests._domains.dataset_comparison.dummy.engine.engine import DummyDatasetComparisonEngine
from tests._domains.dataset_comparison.dummy.resource_spec import DummyDatasetSpec
from tests._domains.dataset_comparison.dummy.resources import DummyDataset
from tests._domains.dataset_comparison.dummy.types.scores import DummyDatasetComparisonScoreType


@pytest.fixture
def ensure_artifact_registration():
    from tests._domains.dataset_comparison.dummy.artifacts.scores.dummy import (
        DummyDatasetComparisonScore,
        DummyDatasetComparisonScoreHyperparams,
    )

    _ = DummyDatasetComparisonScore, DummyDatasetComparisonScoreHyperparams
    yield


@pytest.mark.unit
@pytest.mark.parametrize(
    "score_type, resource_spec, dataset_real, dataset_synthetic, expected_result",
    [
        (
            DummyDatasetComparisonScoreType.DUMMY_SCORE,
            DummyDatasetSpec(scale=1),
            DummyDataset(x=1.0),
            DummyDataset(x=1.0),
            0,
        ),
        (
            DummyDatasetComparisonScoreType.DUMMY_SCORE,
            DummyDatasetSpec(scale=2),
            DummyDataset(x=1.0),
            DummyDataset(x=1.0),
            0,
        ),
        (
            DummyDatasetComparisonScoreType.DUMMY_SCORE,
            DummyDatasetSpec(scale=2),
            DummyDataset(x=3.0),
            DummyDataset(x=1.0),
            4,
        ),
        (
            DummyDatasetComparisonScoreType.DUMMY_SCORE,
            DummyDatasetSpec(scale=1),
            DummyDataset(x=5.0),
            DummyDataset(x=2.0),
            3,
        ),
    ],
)
def test_produce_dataset_comparison_score(
    ensure_artifact_registration,
    score_type: DummyDatasetComparisonScoreType,
    resource_spec: DummyDatasetSpec,
    dataset_real: DummyDataset,
    dataset_synthetic: DummyDataset,
    expected_result: float,
):
    engine = DummyDatasetComparisonEngine.build(resource_spec=resource_spec)
    result = engine.produce_dataset_comparison_score(
        score_type=score_type, dataset_real=dataset_real, dataset_synthetic=dataset_synthetic
    )
    assert result == expected_result


@pytest.mark.unit
@pytest.mark.parametrize(
    "score_type, resource_spec, dataset_real, dataset_synthetic",
    [
        (
            DummyDatasetComparisonScoreType.DUMMY_SCORE,
            DummyDatasetSpec(scale=1),
            DummyDataset(x="invalid"),
            DummyDataset(x=1.0),
        ),
        (
            DummyDatasetComparisonScoreType.DUMMY_SCORE,
            DummyDatasetSpec(scale=2),
            DummyDataset(x=1.0),
            DummyDataset(x="invalid"),
        ),
        (
            DummyDatasetComparisonScoreType.DUMMY_SCORE,
            DummyDatasetSpec(scale=1),
            DummyDataset(x="invalid"),
            DummyDataset(x="invalid"),
        ),
    ],
)
def test_produce_dataset_comparison_score_invalid_datasets(
    ensure_artifact_registration,
    score_type: DummyDatasetComparisonScoreType,
    resource_spec: DummyDatasetSpec,
    dataset_real: DummyDataset,
    dataset_synthetic: DummyDataset,
):
    engine = DummyDatasetComparisonEngine.build(resource_spec=resource_spec)
    with pytest.raises(ValueError, match="Invalid Data: expected float, got"):
        engine.produce_dataset_comparison_score(
            score_type=score_type, dataset_real=dataset_real, dataset_synthetic=dataset_synthetic
        )
