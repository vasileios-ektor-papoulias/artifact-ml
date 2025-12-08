import pytest
from artifact_core._base.core.hyperparams import NO_ARTIFACT_HYPERPARAMS
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
from artifact_core.binary_classification._artifacts.scores.prediction_metrics import (
    AccuracyScore,
)
from artifact_core.binary_classification._resources import BinaryClassificationArtifactResources
from pytest_mock import MockerFixture


@pytest.mark.unit
def test_accuracy_compute(
    mocker: MockerFixture,
    resource_spec: BinaryClassSpecProtocol,
    true_class_store: BinaryClassStore,
    classification_results: BinaryClassificationResults,
):
    fake_score: float = 0.85
    patch_compute = mocker.patch.object(
        target=BinaryPredictionMetricCalculator,
        attribute="compute",
        return_value=fake_score,
    )
    artifact = AccuracyScore(resource_spec=resource_spec, hyperparams=NO_ARTIFACT_HYPERPARAMS)
    resources = BinaryClassificationArtifactResources.from_stores(
        true_class_store=true_class_store,
        classification_results=classification_results,
    )
    result = artifact.compute(resources=resources)
    patch_compute.assert_called_once_with(
        metric_type=BinaryPredictionMetric.ACCURACY,
        true=true_class_store.id_to_is_positive,
        predicted=classification_results.id_to_predicted_positive,
    )
    assert result == fake_score
