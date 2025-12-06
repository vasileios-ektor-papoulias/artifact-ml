import numpy as np
import pytest
from artifact_core._libs.artifacts.binary_classification.confusion.calculator import (
    ConfusionMatrixNormalizationStrategy,
    NormalizedConfusionCalculator,
)
from artifact_core._libs.resource_specs.binary_classification.protocol import (
    BinaryClassSpecProtocol,
)
from artifact_core._libs.resources.binary_classification.class_store import BinaryClassStore
from artifact_core._libs.resources.binary_classification.classification_results import (
    BinaryClassificationResults,
)
from artifact_core.binary_classification._artifacts.arrays.confusion import (
    ConfusionMatrix,
    ConfusionMatrixHyperparams,
)
from artifact_core.binary_classification._resources import BinaryClassificationArtifactResources
from pytest_mock import MockerFixture


@pytest.fixture
def hyperparams() -> ConfusionMatrixHyperparams:
    return ConfusionMatrixHyperparams(normalization=ConfusionMatrixNormalizationStrategy.ALL)


@pytest.mark.unit
def test_compute(
    mocker: MockerFixture,
    resource_spec: BinaryClassSpecProtocol,
    true_class_store: BinaryClassStore,
    classification_results: BinaryClassificationResults,
    hyperparams: ConfusionMatrixHyperparams,
):
    fake_matrix = np.array([[0.4, 0.1], [0.2, 0.3]])
    patch_compute = mocker.patch.object(
        target=NormalizedConfusionCalculator,
        attribute="compute_normalized_confusion_matrix",
        return_value=fake_matrix,
    )
    artifact = ConfusionMatrix(resource_spec=resource_spec, hyperparams=hyperparams)
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
    np.testing.assert_array_equal(result, fake_matrix)
