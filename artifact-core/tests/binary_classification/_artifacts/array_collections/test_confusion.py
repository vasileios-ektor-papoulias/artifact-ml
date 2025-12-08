from typing import Dict

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
from artifact_core.binary_classification._artifacts.array_collections.confusion import (
    ConfusionMatrixCollection,
    ConfusionMatrixCollectionHyperparams,
)
from artifact_core.binary_classification._resources import BinaryClassificationArtifactResources
from pytest_mock import MockerFixture


@pytest.fixture
def hyperparams() -> ConfusionMatrixCollectionHyperparams:
    return ConfusionMatrixCollectionHyperparams(
        normalization_types=[
            ConfusionMatrixNormalizationStrategy.ALL,
            ConfusionMatrixNormalizationStrategy.TRUE,
        ]
    )


@pytest.mark.unit
def test_compute(
    mocker: MockerFixture,
    resource_spec: BinaryClassSpecProtocol,
    true_class_store: BinaryClassStore,
    classification_results: BinaryClassificationResults,
    hyperparams: ConfusionMatrixCollectionHyperparams,
):
    fake_matrices: Dict[ConfusionMatrixNormalizationStrategy, np.ndarray] = {
        ConfusionMatrixNormalizationStrategy.ALL: np.array([[0.4, 0.1], [0.2, 0.3]]),
        ConfusionMatrixNormalizationStrategy.TRUE: np.array([[0.8, 0.2], [0.4, 0.6]]),
    }
    patch_compute = mocker.patch.object(
        target=NormalizedConfusionCalculator,
        attribute="compute_confusion_matrix_multiple_normalizations",
        return_value=fake_matrices,
    )
    artifact = ConfusionMatrixCollection(resource_spec=resource_spec, hyperparams=hyperparams)
    resources = BinaryClassificationArtifactResources.from_stores(
        true_class_store=true_class_store,
        classification_results=classification_results,
    )
    result = artifact.compute(resources=resources)
    patch_compute.assert_called_once_with(
        true=true_class_store.id_to_is_positive,
        predicted=classification_results.id_to_predicted_positive,
        normalization_types=hyperparams.normalization_types,
    )
    expected = {norm_type.value: matrix for norm_type, matrix in fake_matrices.items()}
    for key in expected:
        np.testing.assert_array_equal(result[key], expected[key])
