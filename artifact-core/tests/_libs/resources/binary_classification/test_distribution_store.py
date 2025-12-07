from typing import List, Mapping

import numpy as np
import pytest
from artifact_core._libs.resource_specs.binary_classification.spec import (
    BinaryClassSpec,
)
from artifact_core._libs.resources.binary_classification.distribution_store import (
    BinaryDistributionStore,
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
    store = BinaryDistributionStore.build_empty(class_spec=spec)
    assert store.class_spec == spec
    assert store.n_items == 0
    assert store.label_name == label_name
    assert list(store.class_names) == class_names
    assert store.n_classes == 2
    assert store.positive_class == positive_class


@pytest.mark.unit
@pytest.mark.parametrize(
    "id_to_prob_pos",
    [
        {0: 0.2, 1: 0.8},
        {0: 0.5},
        {0: 0.0, 1: 1.0, 2: 0.5},
    ],
)
def test_build(
    binary_spec_params: BinarySpecParams,
    id_to_prob_pos: Mapping[IdentifierType, float],
):
    store = BinaryDistributionStore.build(
        class_names=list(binary_spec_params.class_names),
        positive_class=binary_spec_params.positive_class,
        label_name=binary_spec_params.label_name,
        id_to_prob_pos=id_to_prob_pos,
    )
    assert store.n_items == len(id_to_prob_pos)
    assert store.positive_class == binary_spec_params.positive_class
    assert store.negative_class == binary_spec_params.negative_class


@pytest.mark.unit
@pytest.mark.parametrize(
    "id_to_prob_pos",
    [
        {0: 0.2, 1: 0.8},
        {0: 0.5},
        {0: 0.0, 1: 1.0, 2: 0.5},
    ],
)
def test_from_spec(
    binary_spec: BinaryClassSpec,
    id_to_prob_pos: Mapping[IdentifierType, float],
):
    store = BinaryDistributionStore.from_spec(
        class_spec=binary_spec, id_to_prob_pos=id_to_prob_pos
    )
    assert store.n_items == len(id_to_prob_pos)
    assert store.class_spec == binary_spec


@pytest.mark.unit
def test_positive_and_negative_class(binary_distribution_store: BinaryDistributionStore):
    assert binary_distribution_store.positive_class == "pos"
    assert binary_distribution_store.negative_class == "neg"


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
def test_set_and_get_prob_pos(
    binary_distribution_store: BinaryDistributionStore,
    identifier: IdentifierType,
    prob_pos: float,
    expected_prob_neg: float,
):
    binary_distribution_store.set_prob_pos(identifier=identifier, prob_pos=prob_pos)
    assert binary_distribution_store.get_prob_pos(identifier=identifier) == pytest.approx(
        expected=prob_pos
    )
    actual_neg = binary_distribution_store.get_prob_neg(identifier=identifier)
    assert actual_neg == pytest.approx(expected=expected_prob_neg)


@pytest.mark.unit
@pytest.mark.parametrize(
    "identifier, logit_pos, expected_prob_pos",
    [
        (0, 0.0, 0.5),
        (1, 100.0, 1.0),
        (2, -100.0, 0.0),
    ],
)
def test_set_and_get_logit_pos(
    binary_distribution_store: BinaryDistributionStore,
    identifier: IdentifierType,
    logit_pos: float,
    expected_prob_pos: float,
):
    binary_distribution_store.set_logit_pos(identifier=identifier, logit_pos=logit_pos)
    actual = binary_distribution_store.get_prob_pos(identifier=identifier)
    assert actual == pytest.approx(expected=expected_prob_pos, abs=1e-5)


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
    binary_distribution_store: BinaryDistributionStore,
    identifier: IdentifierType,
    prob_pos: float,
):
    binary_distribution_store.set_prob_pos(identifier=identifier, prob_pos=prob_pos)
    result_pos, result_neg = binary_distribution_store.get_probs_pos_neg(identifier=identifier)
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
    binary_distribution_store: BinaryDistributionStore,
    identifier: IdentifierType,
    prob_pos: float,
):
    binary_distribution_store.set_prob_pos(identifier=identifier, prob_pos=prob_pos)
    logit_pos, logit_neg = binary_distribution_store.get_logits_pos_neg(identifier=identifier)
    if prob_pos == 0.5:
        assert logit_pos == pytest.approx(expected=logit_neg)
    elif prob_pos > 0.5:
        assert logit_pos > logit_neg
    else:
        assert logit_pos < logit_neg


@pytest.mark.unit
@pytest.mark.parametrize(
    "id_to_prob_pos",
    [
        {0: 0.2, 1: 0.8},
        {0: 0.5},
        {0: 0.0, 1: 1.0, 2: 0.5},
    ],
)
def test_id_to_prob_pos_and_neg(
    binary_distribution_store: BinaryDistributionStore,
    id_to_prob_pos: Mapping[IdentifierType, float],
):
    binary_distribution_store.set_prob_pos_multiple(id_to_prob_pos=id_to_prob_pos)
    result_prob_pos = binary_distribution_store.id_to_prob_pos
    result_prob_neg = binary_distribution_store.id_to_prob_neg
    for identifier, expected in id_to_prob_pos.items():
        actual_pos = result_prob_pos[identifier]
        actual_neg = result_prob_neg[identifier]
        assert actual_pos == pytest.approx(expected=expected)
        assert actual_neg == pytest.approx(expected=1.0 - expected)


@pytest.mark.unit
@pytest.mark.parametrize(
    "id_to_prob_pos",
    [
        {0: 0.5},
        {0: 0.8},
        {0: 0.2, 1: 0.9},
    ],
)
def test_id_to_logit_pos_and_neg(
    binary_distribution_store: BinaryDistributionStore,
    id_to_prob_pos: Mapping[IdentifierType, float],
):
    binary_distribution_store.set_prob_pos_multiple(id_to_prob_pos=id_to_prob_pos)
    logit_pos = binary_distribution_store.id_to_logit_pos
    logit_neg = binary_distribution_store.id_to_logit_neg
    for identifier, prob_pos in id_to_prob_pos.items():
        if prob_pos == 0.5:
            assert logit_pos[identifier] == pytest.approx(expected=logit_neg[identifier])
        elif prob_pos > 0.5:
            assert logit_pos[identifier] > logit_neg[identifier]
        else:
            assert logit_pos[identifier] < logit_neg[identifier]


@pytest.mark.unit
@pytest.mark.parametrize(
    "id_to_prob_pos",
    [
        {0: 0.2, 1: 0.8},
        {0: 0.5},
        {0: 0.0, 1: 1.0, 2: 0.5},
    ],
)
def test_set_prob_pos_multiple(
    binary_distribution_store: BinaryDistributionStore,
    id_to_prob_pos: Mapping[IdentifierType, float],
):
    binary_distribution_store.set_prob_pos_multiple(id_to_prob_pos=id_to_prob_pos)
    assert binary_distribution_store.n_items == len(id_to_prob_pos)
    for identifier, expected in id_to_prob_pos.items():
        actual = binary_distribution_store.get_prob_pos(identifier=identifier)
        assert actual == pytest.approx(expected=expected)


@pytest.mark.unit
@pytest.mark.parametrize(
    "id_to_logit_pos, expected_prob_at_zero",
    [
        ({0: 0.0, 1: 1.0}, 0.5),
        ({0: 0.0}, 0.5),
        ({0: 100.0}, 1.0),
        ({0: -100.0}, 0.0),
    ],
)
def test_set_logit_pos_multiple(
    binary_distribution_store: BinaryDistributionStore,
    id_to_logit_pos: Mapping[IdentifierType, float],
    expected_prob_at_zero: float,
):
    binary_distribution_store.set_logit_pos_multiple(id_to_logit_pos=id_to_logit_pos)
    assert binary_distribution_store.n_items == len(id_to_logit_pos)
    actual = binary_distribution_store.get_prob_pos(identifier=0)
    assert actual == pytest.approx(expected=expected_prob_at_zero, abs=1e-5)


@pytest.mark.unit
@pytest.mark.parametrize(
    "id_to_prob_pos",
    [
        {0: 0.2, 1: 0.8},
        {0: 0.5, 1: 0.5, 2: 0.5},
    ],
)
def test_inherits_distribution_store_methods(
    binary_distribution_store: BinaryDistributionStore,
    id_to_prob_pos: Mapping[IdentifierType, float],
):
    binary_distribution_store.set_prob_pos_multiple(id_to_prob_pos=id_to_prob_pos)
    first_id = next(iter(id_to_prob_pos.keys()))

    logits = binary_distribution_store.get_logits(identifier=first_id)
    assert logits.shape == (2,)

    probs = binary_distribution_store.get_probs(identifier=first_id)
    assert probs.shape == (2,)
    np.testing.assert_almost_equal(actual=probs.sum(), desired=1.0)

    assert binary_distribution_store.n_classes == 2
