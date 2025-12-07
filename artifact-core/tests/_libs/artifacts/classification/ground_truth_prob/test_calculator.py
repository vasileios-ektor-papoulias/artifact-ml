from typing import Dict

import numpy as np
import pytest
from artifact_core._libs.artifacts.classification.ground_truth_prob.calculator import (
    GroundTruthProbCalculator,
)
from artifact_core._libs.resource_specs.classification.spec import ClassSpec
from artifact_core._libs.resources.classification.class_store import ClassStore
from artifact_core._libs.resources.classification.classification_results import (
    ClassificationResults,
)
from artifact_core._utils.collections.entity_store import IdentifierType


@pytest.mark.unit
@pytest.mark.parametrize(
    "true_class, logits",
    [
        ("A", np.array([2.0, 0.5, 0.1])),
        ("B", np.array([0.1, 2.5, 0.2])),
        ("C", np.array([0.2, 0.1, 3.0])),
    ],
)
def test_compute_ground_truth_prob(
    class_spec: ClassSpec,
    true_class_store: ClassStore,
    classification_results: ClassificationResults,
    true_class: str,
    logits: np.ndarray,
):
    identifier = 0
    true_class_store.set_class(identifier=identifier, class_name=true_class)
    classification_results.set_single(
        identifier=identifier, predicted_class=true_class, logits=logits
    )
    result = GroundTruthProbCalculator.compute_ground_truth_prob(
        classification_results=classification_results,
        true_class_store=true_class_store,
        identifier=identifier,
    )
    true_idx = class_spec.get_class_idx(class_name=true_class)
    probs = classification_results.get_probs(identifier=identifier)
    assert result == pytest.approx(expected=probs[true_idx])
    assert 0.0 <= result <= 1.0


@pytest.mark.unit
@pytest.mark.parametrize(
    "true_class, logits",
    [
        ("A", np.array([2.0, 0.5, 0.1])),
        ("B", np.array([0.1, 2.5, 0.2])),
        ("C", np.array([0.2, 0.1, 3.0])),
    ],
)
def test_compute_ground_truth_logit(
    class_spec: ClassSpec,
    true_class_store: ClassStore,
    classification_results: ClassificationResults,
    true_class: str,
    logits: np.ndarray,
):
    identifier = 0
    true_class_store.set_class(identifier=identifier, class_name=true_class)
    classification_results.set_single(
        identifier=identifier, predicted_class=true_class, logits=logits
    )
    result = GroundTruthProbCalculator.compute_ground_truth_logit(
        classification_results=classification_results,
        true_class_store=true_class_store,
        identifier=identifier,
    )
    true_idx = class_spec.get_class_idx(class_name=true_class)
    assert result == pytest.approx(expected=logits[true_idx])


@pytest.mark.unit
@pytest.mark.parametrize(
    "id_to_true_class, id_to_logits",
    [
        (
            {0: "A", 1: "B", 2: "C"},
            {
                0: np.array([2.0, 0.5, 0.1]),
                1: np.array([0.1, 2.5, 0.2]),
                2: np.array([0.2, 0.1, 3.0]),
            },
        ),
        (
            {0: "A"},
            {0: np.array([1.0, 0.5, 0.5])},
        ),
        (
            {10: "B", 20: "C"},
            {
                10: np.array([0.1, 2.0, 0.1]),
                20: np.array([0.1, 0.1, 2.0]),
            },
        ),
    ],
)
def test_compute_id_to_prob_ground_truth(
    true_class_store: ClassStore,
    classification_results: ClassificationResults,
    id_to_true_class: Dict[IdentifierType, str],
    id_to_logits: Dict[IdentifierType, np.ndarray],
):
    for identifier, true_class in id_to_true_class.items():
        true_class_store.set_class(identifier=identifier, class_name=true_class)
        classification_results.set_single(
            identifier=identifier,
            predicted_class=true_class,
            logits=id_to_logits[identifier],
        )
    result = GroundTruthProbCalculator.compute_id_to_prob_ground_truth(
        classification_results=classification_results,
        true_class_store=true_class_store,
    )
    assert set(result.keys()) == set(id_to_true_class.keys())
    for identifier, prob in result.items():
        assert 0.0 <= prob <= 1.0


@pytest.mark.unit
@pytest.mark.parametrize(
    "id_to_true_class, id_to_logits",
    [
        (
            {0: "A", 1: "B", 2: "C"},
            {
                0: np.array([2.0, 0.5, 0.1]),
                1: np.array([0.1, 2.5, 0.2]),
                2: np.array([0.2, 0.1, 3.0]),
            },
        ),
        (
            {0: "A"},
            {0: np.array([1.0, 0.5, 0.5])},
        ),
    ],
)
def test_compute_id_to_logit_ground_truth(
    class_spec: ClassSpec,
    true_class_store: ClassStore,
    classification_results: ClassificationResults,
    id_to_true_class: Dict[IdentifierType, str],
    id_to_logits: Dict[IdentifierType, np.ndarray],
):
    for identifier, true_class in id_to_true_class.items():
        true_class_store.set_class(identifier=identifier, class_name=true_class)
        classification_results.set_single(
            identifier=identifier,
            predicted_class=true_class,
            logits=id_to_logits[identifier],
        )
    result = GroundTruthProbCalculator.compute_id_to_logit_ground_truth(
        classification_results=classification_results,
        true_class_store=true_class_store,
    )
    assert set(result.keys()) == set(id_to_true_class.keys())
    for identifier, logit in result.items():
        true_class = id_to_true_class[identifier]
        true_idx = class_spec.get_class_idx(class_name=true_class)
        expected_logit = id_to_logits[identifier][true_idx]
        assert logit == pytest.approx(expected=expected_logit)


@pytest.mark.unit
def test_compute_raises_on_no_common_ids(
    true_class_store: ClassStore, classification_results: ClassificationResults
):
    true_class_store.set_class(identifier=0, class_name="A")
    classification_results.set_single(identifier=1, predicted_class="A")
    with pytest.raises(KeyError, match="No common ids"):
        GroundTruthProbCalculator.compute_id_to_prob_ground_truth(
            classification_results=classification_results,
            true_class_store=true_class_store,
        )
