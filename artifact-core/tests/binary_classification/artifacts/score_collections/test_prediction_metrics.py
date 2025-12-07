from typing import Dict

import pytest
from artifact_core._libs.artifacts.binary_classification.prediction_metrics.calculator import (
    BinaryPredictionMetric,
    BinaryPredictionMetricCalculator,
)
from artifact_core._libs.resource_specs.binary_classification.protocol import (
    BinaryClassSpecProtocol,
)
from artifact_core._libs.resources.binary_classification.class_store import BinaryClassStore
from artifact_core._libs.resources.binary_classification.classification_results import (
    BinaryClassificationResults,
)
from artifact_core.binary_classification._artifacts.score_collections.prediction_metrics import (
    BinaryPredictionScores,
    BinaryPredictionScoresHyperparams,
)
from artifact_core.binary_classification._resources import BinaryClassificationArtifactResources
from pytest_mock import MockerFixture


@pytest.fixture
def hyperparams() -> BinaryPredictionScoresHyperparams:
    return BinaryPredictionScoresHyperparams(
        metric_types=[BinaryPredictionMetric.ACCURACY, BinaryPredictionMetric.PRECISION]
    )


@pytest.mark.unit
def test_compute(
    mocker: MockerFixture,
    resource_spec: BinaryClassSpecProtocol,
    true_class_store: BinaryClassStore,
    classification_results: BinaryClassificationResults,
    hyperparams: BinaryPredictionScoresHyperparams,
):
    fake_scores: Dict[BinaryPredictionMetric, float] = {
        BinaryPredictionMetric.ACCURACY: 0.85,
        BinaryPredictionMetric.PRECISION: 0.75,
    }
    patch_compute = mocker.patch.object(
        target=BinaryPredictionMetricCalculator,
        attribute="compute_multiple",
        return_value=fake_scores,
    )
    artifact = BinaryPredictionScores(resource_spec=resource_spec, hyperparams=hyperparams)
    resources = BinaryClassificationArtifactResources.from_stores(
        true_class_store=true_class_store,
        classification_results=classification_results,
    )
    result = artifact.compute(resources=resources)
    patch_compute.assert_called_once_with(
        metric_types=hyperparams.metric_types,
        true=true_class_store.id_to_is_positive,
        predicted=classification_results.id_to_predicted_positive,
    )
    expected = {
        metric_type.value: metric_value for metric_type, metric_value in fake_scores.items()
    }
    assert result == expected
