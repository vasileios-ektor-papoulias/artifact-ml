from typing import Dict

import pytest
from artifact_core._libs.artifacts.binary_classification.confusion.calculator import (
    ConfusionMatrixNormalizationStrategy,
    NormalizedConfusionCalculator,
)
from artifact_core._libs.artifacts.binary_classification.confusion.raw import (
    ConfusionMatrixCell,
)
from artifact_core._libs.resource_specs.binary_classification.protocol import (
    BinaryClassSpecProtocol,
)
from artifact_core._libs.resources.binary_classification.class_store import BinaryClassStore
from artifact_core._libs.resources.binary_classification.classification_results import (
    BinaryClassificationResults,
)
from artifact_core.binary_classification._artifacts.score_collections.confusion import (
    NormalizedConfusionCounts,
    NormalizedConfusionCountsHyperparams,
)
from artifact_core.binary_classification._resources import BinaryClassificationArtifactResources
from pytest_mock import MockerFixture


@pytest.fixture
def hyperparams() -> NormalizedConfusionCountsHyperparams:
    return NormalizedConfusionCountsHyperparams(
        normalization=ConfusionMatrixNormalizationStrategy.ALL
    )


@pytest.mark.unit
def test_compute(
    mocker: MockerFixture,
    resource_spec: BinaryClassSpecProtocol,
    true_class_store: BinaryClassStore,
    classification_results: BinaryClassificationResults,
    hyperparams: NormalizedConfusionCountsHyperparams,
):
    fake_scores: Dict[ConfusionMatrixCell, float] = {
        ConfusionMatrixCell.TRUE_POSITIVE: 0.4,
        ConfusionMatrixCell.TRUE_NEGATIVE: 0.3,
        ConfusionMatrixCell.FALSE_POSITIVE: 0.2,
        ConfusionMatrixCell.FALSE_NEGATIVE: 0.1,
    }
    patch_compute = mocker.patch.object(
        target=NormalizedConfusionCalculator,
        attribute="compute_dict_normalized_confusion_counts",
        return_value=fake_scores,
    )
    artifact = NormalizedConfusionCounts(resource_spec=resource_spec, hyperparams=hyperparams)
    resources = BinaryClassificationArtifactResources.from_stores(
        true_class_store=true_class_store,
        classification_results=classification_results,
    )
    result = artifact.compute(resources=resources)
    patch_compute.assert_called_once_with(
        true=true_class_store.id_to_is_positive,
        predicted=classification_results.id_to_predicted_positive,
        normalization=hyperparams.normalization,
    )
    expected = {
        metric_type.value: metric_value for metric_type, metric_value in fake_scores.items()
    }
    assert result == expected
