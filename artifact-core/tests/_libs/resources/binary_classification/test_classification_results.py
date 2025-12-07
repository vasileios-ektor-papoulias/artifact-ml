from typing import List, Mapping

import pytest
from artifact_core._libs.resource_specs.binary_classification.spec import (
    BinaryClassSpec,
)
from artifact_core._libs.resources.binary_classification.classification_results import (
    BinaryClassificationResults,
)
from artifact_core._utils.collections.entity_store import IdentifierType

from tests._libs.resources.binary_classification.conftest import BinarySpecParams


@pytest.mark.unit
@pytest.mark.parametrize(
    "class_names, positive_class, label_name",
    [
        (["neg", "pos"], "pos", "target"),
        (["0", "1"], "1", "label"),
        (["no", "yes"], "yes", "result"),
    ],
)
def test_build_empty(class_names: List[str], positive_class: str, label_name: str):
    spec = BinaryClassSpec(
        class_names=class_names, positive_class=positive_class, label_name=label_name
    )
    results = BinaryClassificationResults.build_empty(class_spec=spec)
    assert results.n_items == 0
    assert results.label_name == label_name
    assert list(results.class_names) == class_names
    assert results.n_classes == 2
    assert results.positive_class == positive_class
    assert results.pred_store is not None
    assert results.distn_store is not None


@pytest.mark.unit
@pytest.mark.parametrize(
    "id_to_class, id_to_prob_pos",
    [
        ({0: "neg", 1: "pos"}, {0: 0.2, 1: 0.8}),
        ({0: "pos"}, {0: 0.9}),
        ({0: "neg", 1: "pos", 2: "pos"}, {0: 0.1, 1: 0.8, 2: 0.9}),
    ],
)
def test_build(
    binary_spec_params: BinarySpecParams,
    id_to_class: Mapping[IdentifierType, str],
    id_to_prob_pos: Mapping[IdentifierType, float],
):
    results = BinaryClassificationResults.build(
        class_names=list(binary_spec_params.class_names),
        positive_class=binary_spec_params.positive_class,
        id_to_class=id_to_class,
        id_to_prob_pos=id_to_prob_pos,
    )
    assert results.n_items == len(id_to_class)
    assert results.positive_class == binary_spec_params.positive_class
    assert results.negative_class == binary_spec_params.negative_class


@pytest.mark.unit
@pytest.mark.parametrize(
    "id_to_class, id_to_prob_pos",
    [
        ({0: "neg", 1: "pos"}, {0: 0.2, 1: 0.8}),
        ({0: "pos"}, {0: 0.9}),
        ({0: "neg", 1: "pos", 2: "pos"}, {0: 0.1, 1: 0.8, 2: 0.9}),
    ],
)
def test_from_spec(
    binary_spec: BinaryClassSpec,
    id_to_class: Mapping[IdentifierType, str],
    id_to_prob_pos: Mapping[IdentifierType, float],
):
    results = BinaryClassificationResults.from_spec(
        class_spec=binary_spec,
        id_to_class=id_to_class,
        id_to_prob_pos=id_to_prob_pos,
    )
    assert results.n_items == len(id_to_class)
    assert results.class_spec == binary_spec


@pytest.mark.unit
@pytest.mark.parametrize(
    "id_to_class, expected_probs",
    [
        ({0: "pos", 1: "neg"}, {0: 1.0, 1: 0.0}),
        ({0: "neg"}, {0: 0.0}),
        ({0: "pos", 1: "pos", 2: "neg"}, {0: 1.0, 1: 1.0, 2: 0.0}),
    ],
)
def test_from_spec_without_probs(
    binary_spec: BinaryClassSpec,
    id_to_class: Mapping[IdentifierType, str],
    expected_probs: Mapping[IdentifierType, float],
):
    results = BinaryClassificationResults.from_spec(
        class_spec=binary_spec, id_to_class=id_to_class
    )
    assert results.n_items == len(id_to_class)
    for identifier, expected_prob in expected_probs.items():
        actual = results.get_prob_pos(identifier=identifier)
        assert actual == pytest.approx(expected=expected_prob)


@pytest.mark.unit
def test_positive_and_negative_class(
    binary_classification_results: BinaryClassificationResults,
):
    assert binary_classification_results.positive_class == "pos"
    assert binary_classification_results.negative_class == "neg"


@pytest.mark.unit
@pytest.mark.parametrize(
    "identifier, class_name, expected_is_positive",
    [
        (0, "pos", True),
        (1, "neg", False),
        (2, "pos", True),
    ],
)
def test_id_to_predicted_positive_and_negative(
    binary_classification_results: BinaryClassificationResults,
    identifier: IdentifierType,
    class_name: str,
    expected_is_positive: bool,
):
    binary_classification_results.set_single_binary(
        identifier=identifier, predicted_class=class_name
    )
    actual_pos = binary_classification_results.id_to_predicted_positive[identifier]
    actual_neg = binary_classification_results.id_to_predicted_negative[identifier]
    assert actual_pos is expected_is_positive
    assert actual_neg is (not expected_is_positive)


@pytest.mark.unit
@pytest.mark.parametrize(
    "id_to_class, id_to_prob_pos",
    [
        ({0: "neg", 1: "pos"}, {0: 0.2, 1: 0.8}),
        ({0: "pos"}, {0: 0.5}),
        ({0: "neg", 1: "pos", 2: "neg"}, {0: 0.1, 1: 0.9, 2: 0.3}),
    ],
)
def test_id_to_prob_pos_and_neg(
    binary_classification_results: BinaryClassificationResults,
    id_to_class: Mapping[IdentifierType, str],
    id_to_prob_pos: Mapping[IdentifierType, float],
):
    binary_classification_results.set_multiple_binary(
        id_to_class=id_to_class, id_to_prob_pos=id_to_prob_pos
    )
    result_prob_pos = binary_classification_results.id_to_prob_pos
    result_prob_neg = binary_classification_results.id_to_prob_neg
    for identifier, expected_prob_pos in id_to_prob_pos.items():
        actual_pos = result_prob_pos[identifier]
        actual_neg = result_prob_neg[identifier]
        assert actual_pos == pytest.approx(expected=expected_prob_pos)
        assert actual_neg == pytest.approx(expected=1.0 - expected_prob_pos)


@pytest.mark.unit
@pytest.mark.parametrize(
    "id_to_class, id_to_prob_pos",
    [
        ({0: "pos"}, {0: 0.5}),
        ({0: "neg"}, {0: 0.5}),
        ({0: "pos"}, {0: 0.8}),
        ({0: "neg", 1: "pos"}, {0: 0.2, 1: 0.9}),
    ],
)
def test_id_to_logit_pos_and_neg(
    binary_classification_results: BinaryClassificationResults,
    id_to_class: Mapping[IdentifierType, str],
    id_to_prob_pos: Mapping[IdentifierType, float],
):
    binary_classification_results.set_multiple_binary(
        id_to_class=id_to_class, id_to_prob_pos=id_to_prob_pos
    )
    logit_pos = binary_classification_results.id_to_logit_pos
    logit_neg = binary_classification_results.id_to_logit_neg
    for identifier, prob_pos in id_to_prob_pos.items():
        if prob_pos == 0.5:
            assert logit_pos[identifier] == pytest.approx(expected=logit_neg[identifier])
        elif prob_pos > 0.5:
            assert logit_pos[identifier] > logit_neg[identifier]
        else:
            assert logit_pos[identifier] < logit_neg[identifier]


@pytest.mark.unit
@pytest.mark.parametrize(
    "class_name, expected_positive, expected_negative",
    [
        ("neg", False, True),
        ("pos", True, False),
    ],
)
def test_predicted_class_is_positive_and_negative(
    binary_classification_results: BinaryClassificationResults,
    class_name: str,
    expected_positive: bool,
    expected_negative: bool,
):
    binary_classification_results.set_single_binary(identifier=0, predicted_class=class_name)
    actual_pos = binary_classification_results.predicted_class_is_positive(identifier=0)
    actual_neg = binary_classification_results.predicted_class_is_negative(identifier=0)
    assert actual_pos is expected_positive
    assert actual_neg is expected_negative


@pytest.mark.unit
@pytest.mark.parametrize(
    "identifier, prob_pos, expected_prob_neg",
    [
        (0, 0.75, 0.25),
        (1, 0.0, 1.0),
        (2, 1.0, 0.0),
        (99, 0.5, 0.5),
    ],
)
def test_get_prob_pos_and_neg(
    binary_classification_results: BinaryClassificationResults,
    identifier: IdentifierType,
    prob_pos: float,
    expected_prob_neg: float,
):
    binary_classification_results.set_single_binary(
        identifier=identifier, predicted_class="pos", prob_pos=prob_pos
    )
    actual_pos = binary_classification_results.get_prob_pos(identifier=identifier)
    actual_neg = binary_classification_results.get_prob_neg(identifier=identifier)
    assert actual_pos == pytest.approx(expected=prob_pos)
    assert actual_neg == pytest.approx(expected=expected_prob_neg)


@pytest.mark.unit
@pytest.mark.parametrize(
    "identifier, prob_pos",
    [
        (0, 0.5),
        (1, 0.8),
        (2, 0.2),
        (3, 0.99),
        (4, 0.01),
    ],
)
def test_get_logit_pos_and_neg(
    binary_classification_results: BinaryClassificationResults,
    identifier: IdentifierType,
    prob_pos: float,
):
    binary_classification_results.set_single_binary(
        identifier=identifier, predicted_class="pos", prob_pos=prob_pos
    )
    logit_pos = binary_classification_results.get_logit_pos(identifier=identifier)
    logit_neg = binary_classification_results.get_logit_neg(identifier=identifier)
    if prob_pos == 0.5:
        assert logit_pos == pytest.approx(expected=logit_neg)
    elif prob_pos > 0.5:
        assert logit_pos > logit_neg
    else:
        assert logit_pos < logit_neg


@pytest.mark.unit
@pytest.mark.parametrize(
    "identifier, prob_pos",
    [
        (0, 0.6),
        (1, 0.3),
        (2, 0.5),
        (3, 0.99),
        (4, 0.01),
    ],
)
def test_get_probs_pos_neg(
    binary_classification_results: BinaryClassificationResults,
    identifier: IdentifierType,
    prob_pos: float,
):
    binary_classification_results.set_single_binary(
        identifier=identifier, predicted_class="pos", prob_pos=prob_pos
    )
    result_pos, result_neg = binary_classification_results.get_probs_pos_neg(identifier=identifier)
    assert result_pos == pytest.approx(expected=prob_pos)
    assert result_neg == pytest.approx(expected=1.0 - prob_pos)


@pytest.mark.unit
@pytest.mark.parametrize(
    "identifier, prob_pos",
    [
        (0, 0.5),
        (1, 0.8),
        (2, 0.2),
        (3, 0.99),
        (4, 0.01),
    ],
)
def test_get_logits_pos_neg(
    binary_classification_results: BinaryClassificationResults,
    identifier: IdentifierType,
    prob_pos: float,
):
    binary_classification_results.set_single_binary(
        identifier=identifier, predicted_class="pos", prob_pos=prob_pos
    )
    logit_pos, logit_neg = binary_classification_results.get_logits_pos_neg(identifier=identifier)
    if prob_pos == 0.5:
        assert logit_pos == pytest.approx(expected=logit_neg)
    elif prob_pos > 0.5:
        assert logit_pos > logit_neg
    else:
        assert logit_pos < logit_neg


@pytest.mark.unit
@pytest.mark.parametrize(
    "class_name, prob_pos, expected_prob",
    [
        ("pos", 0.9, 0.9),
        ("neg", 0.1, 0.1),
        ("pos", None, 1.0),
        ("neg", None, 0.0),
    ],
)
def test_set_single_binary(
    binary_classification_results: BinaryClassificationResults,
    class_name: str,
    prob_pos: float | None,
    expected_prob: float,
):
    if prob_pos is not None:
        binary_classification_results.set_single_binary(
            identifier=0, predicted_class=class_name, prob_pos=prob_pos
        )
    else:
        binary_classification_results.set_single_binary(
            identifier=0, predicted_class=class_name
        )
    assert binary_classification_results.get_predicted_class(identifier=0) == class_name
    actual_prob = binary_classification_results.get_prob_pos(identifier=0)
    assert actual_prob == pytest.approx(expected=expected_prob)


@pytest.mark.unit
@pytest.mark.parametrize(
    "id_to_class, id_to_prob_pos",
    [
        ({0: "neg", 1: "pos"}, {0: 0.2, 1: 0.8}),
        ({0: "pos"}, {0: 0.9}),
        ({0: "neg", 1: "pos", 2: "neg"}, {0: 0.1, 1: 0.8, 2: 0.3}),
    ],
)
def test_set_multiple_binary(
    binary_classification_results: BinaryClassificationResults,
    id_to_class: Mapping[IdentifierType, str],
    id_to_prob_pos: Mapping[IdentifierType, float],
):
    binary_classification_results.set_multiple_binary(
        id_to_class=id_to_class, id_to_prob_pos=id_to_prob_pos
    )
    assert binary_classification_results.n_items == len(id_to_class)
    for identifier, expected_class in id_to_class.items():
        actual_class = binary_classification_results.get_predicted_class(identifier=identifier)
        assert actual_class == expected_class
    for identifier, expected_prob in id_to_prob_pos.items():
        actual_prob = binary_classification_results.get_prob_pos(identifier=identifier)
        assert actual_prob == pytest.approx(expected=expected_prob)


@pytest.mark.unit
@pytest.mark.parametrize(
    "id_to_class",
    [
        {0: "neg", 1: "pos"},
        {0: "pos", 1: "neg", 2: "pos"},
    ],
)
def test_inherits_classification_results(
    binary_classification_results: BinaryClassificationResults,
    id_to_class: Mapping[IdentifierType, str],
):
    binary_classification_results.set_multiple_binary(id_to_class=id_to_class)
    assert binary_classification_results.n_classes == 2
    assert binary_classification_results.pred_store is not None
    assert binary_classification_results.distn_store is not None
