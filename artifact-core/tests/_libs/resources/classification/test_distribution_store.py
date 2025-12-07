from typing import List, Mapping

import numpy as np
import pytest
from artifact_core._libs.resource_specs.classification.spec import ClassSpec
from artifact_core._libs.resources.classification.distribution_store import (
    ClassDistributionStore,
)
from artifact_core._utils.collections.entity_store import IdentifierType


@pytest.mark.unit
@pytest.mark.parametrize(
    "class_names, label_name",
    [
        (["A", "B", "C"], "target"),
        (["neg", "pos"], "label"),
        (["cat", "dog", "bird", "fish"], "animal"),
    ],
)
def test_init_empty(class_names: List[str], label_name: str):
    spec = ClassSpec(class_names=class_names, label_name=label_name)
    store = ClassDistributionStore(class_spec=spec)
    assert store.class_spec == spec
    assert store.n_items == 0
    assert store.label_name == label_name
    assert list(store.class_names) == class_names
    assert store.n_classes == len(class_names)


@pytest.mark.unit
@pytest.mark.parametrize(
    "class_names, label_name",
    [
        (["A", "B", "C"], "target"),
        (["neg", "pos"], "label"),
        (["cat", "dog", "bird", "fish"], "animal"),
    ],
)
def test_build_empty(class_names: List[str], label_name: str):
    spec = ClassSpec(class_names=class_names, label_name=label_name)
    store = ClassDistributionStore.build_empty(class_spec=spec)
    assert store.class_spec == spec
    assert store.n_items == 0
    assert store.label_name == label_name
    assert list(store.class_names) == class_names
    assert store.n_classes == len(class_names)


@pytest.mark.unit
@pytest.mark.parametrize(
    "class_names, label_name, id_to_logits",
    [
        (["A", "B", "C"], "target", {0: [1.0, 2.0, 3.0], 1: [0.0, 0.0, 0.0]}),
        (["neg", "pos"], "label", {0: [1.0, 2.0]}),
    ],
)
def test_init_with_logits(
    class_names: List[str], label_name: str, id_to_logits: Mapping[IdentifierType, List[float]]
):
    spec = ClassSpec(class_names=class_names, label_name=label_name)
    logits = {k: np.array(v) for k, v in id_to_logits.items()}
    store = ClassDistributionStore(class_spec=spec, id_to_logits=logits)
    assert store.n_items == len(logits)


@pytest.mark.unit
@pytest.mark.parametrize(
    "identifier, logits",
    [
        (0, [1.0, 2.0, 3.0]),
        (1, [0.0, 0.0, 0.0]),
        (99, [-1.0, 0.0, 1.0]),
    ],
)
def test_set_and_get_logits(
    distribution_store: ClassDistributionStore, identifier: IdentifierType, logits: List[float]
):
    arr_logits = np.array(logits)
    distribution_store.set_logits(identifier=identifier, logits=arr_logits)
    retrieved = distribution_store.get_logits(identifier=identifier)
    np.testing.assert_array_almost_equal(retrieved, arr_logits)


@pytest.mark.unit
@pytest.mark.parametrize(
    "identifier, probs",
    [
        (0, [0.2, 0.3, 0.5]),
        (1, [0.33, 0.33, 0.34]),
        (99, [1.0, 0.0, 0.0]),
    ],
)
def test_set_and_get_probs(
    distribution_store: ClassDistributionStore, identifier: IdentifierType, probs: List[float]
):
    arr_probs = np.array(probs)
    distribution_store.set_probs(identifier=identifier, probs=arr_probs)
    retrieved = distribution_store.get_probs(identifier=identifier)
    np.testing.assert_array_almost_equal(retrieved, arr_probs)


@pytest.mark.unit
@pytest.mark.parametrize(
    "identifier, logits, class_name, class_idx",
    [
        (0, [1.0, 2.0, 3.0], "A", 0),
        (0, [1.0, 2.0, 3.0], "B", 1),
        (0, [1.0, 2.0, 3.0], "C", 2),
        (5, [0.5, 1.5, 2.5], "B", 1),
    ],
)
def test_get_logit_by_class(
    distribution_store: ClassDistributionStore,
    identifier: IdentifierType,
    logits: List[float],
    class_name: str,
    class_idx: int,
):
    arr_logits = np.array(logits)
    distribution_store.set_logits(identifier=identifier, logits=arr_logits)
    actual = distribution_store.get_logit(class_name=class_name, identifier=identifier)
    assert actual == arr_logits[class_idx]


@pytest.mark.unit
@pytest.mark.parametrize(
    "identifier, logits",
    [
        (0, [1.0, 2.0, 3.0]),
        (5, [0.0, 0.0, 0.0]),
    ],
)
def test_get_prob_by_class(
    distribution_store: ClassDistributionStore, identifier: IdentifierType, logits: List[float]
):
    arr_logits = np.array(logits)
    distribution_store.set_logits(identifier=identifier, logits=arr_logits)
    probs = distribution_store.get_probs(identifier=identifier)
    for idx, class_name in enumerate(distribution_store.class_names):
        prob = distribution_store.get_prob(class_name=class_name, identifier=identifier)
        assert prob == pytest.approx(expected=probs[idx])


@pytest.mark.unit
@pytest.mark.parametrize(
    "identifier, class_name, expected_probs",
    [
        (0, "A", [1.0, 0.0, 0.0]),
        (0, "B", [0.0, 1.0, 0.0]),
        (0, "C", [0.0, 0.0, 1.0]),
        (99, "A", [1.0, 0.0, 0.0]),
    ],
)
def test_set_concentrated(
    distribution_store: ClassDistributionStore,
    identifier: IdentifierType,
    class_name: str,
    expected_probs: List[float],
):
    distribution_store.set_concentrated(identifier=identifier, class_name=class_name)
    probs = distribution_store.get_probs(identifier=identifier)
    np.testing.assert_array_almost_equal(probs, expected_probs)


@pytest.mark.unit
@pytest.mark.parametrize(
    "identifier, class_idx, expected_probs",
    [
        (0, 0, [1.0, 0.0, 0.0]),
        (0, 1, [0.0, 1.0, 0.0]),
        (0, 2, [0.0, 0.0, 1.0]),
        (99, 1, [0.0, 1.0, 0.0]),
    ],
)
def test_set_concentrated_idx(
    distribution_store: ClassDistributionStore,
    identifier: IdentifierType,
    class_idx: int,
    expected_probs: List[float],
):
    distribution_store.set_concentrated_idx(identifier=identifier, class_idx=class_idx)
    probs = distribution_store.get_probs(identifier=identifier)
    np.testing.assert_array_almost_equal(probs, expected_probs)


@pytest.mark.unit
@pytest.mark.parametrize(
    "id_to_logits",
    [
        {0: [1.0, 2.0, 3.0], 1: [0.0, 0.0, 0.0]},
        {0: [1.0, 2.0, 3.0]},
        {10: [0.5, 0.5, 0.5], 20: [1.0, 1.0, 1.0], 30: [0.0, 0.0, 0.0]},
    ],
)
def test_arr_logits_and_probs(
    distribution_store: ClassDistributionStore, id_to_logits: Mapping[IdentifierType, List[float]]
):
    for identifier, logits in id_to_logits.items():
        distribution_store.set_logits(identifier=identifier, logits=np.array(logits))

    arr_logits = distribution_store.arr_logits
    assert arr_logits.shape == (len(id_to_logits), distribution_store.n_classes)

    arr_probs = distribution_store.arr_probs
    assert arr_probs.shape == (len(id_to_logits), distribution_store.n_classes)
    expected_sums = np.ones(len(id_to_logits))
    np.testing.assert_array_almost_equal(arr_probs.sum(axis=1), expected_sums)


@pytest.mark.unit
def test_arr_logits_empty(distribution_store: ClassDistributionStore):
    arr_logits = distribution_store.arr_logits
    assert arr_logits.shape == (0, distribution_store.n_classes)


@pytest.mark.unit
@pytest.mark.parametrize(
    "identifier, logits",
    [
        (0, [1.0, 2.0, 3.0]),
        (5, [-1.0, 0.0, 1.0]),
    ],
)
def test_id_to_logits_and_probs(
    distribution_store: ClassDistributionStore, identifier: IdentifierType, logits: List[float]
):
    arr_logits = np.array(logits)
    distribution_store.set_logits(identifier=identifier, logits=arr_logits)

    id_to_logits = distribution_store.id_to_logits
    assert identifier in id_to_logits
    np.testing.assert_array_almost_equal(id_to_logits[identifier], arr_logits)

    id_to_probs = distribution_store.id_to_probs
    assert identifier in id_to_probs
    assert pytest.approx(expected=sum(id_to_probs[identifier])) == 1.0


@pytest.mark.unit
@pytest.mark.parametrize(
    "id_to_logits",
    [
        {0: [1.0, 2.0, 3.0], 1: [0.0, 0.0, 0.0]},
        {10: [0.5, 0.5, 0.5]},
    ],
)
def test_set_logits_multiple(
    distribution_store: ClassDistributionStore, id_to_logits: Mapping[IdentifierType, List[float]]
):
    logits_data = {k: np.array(v) for k, v in id_to_logits.items()}
    distribution_store.set_logits_multiple(id_to_logits=logits_data)
    assert distribution_store.n_items == len(logits_data)


@pytest.mark.unit
@pytest.mark.parametrize(
    "id_to_probs",
    [
        {0: [0.2, 0.3, 0.5], 1: [0.1, 0.2, 0.7]},
        {10: [0.33, 0.33, 0.34]},
    ],
)
def test_set_probs_multiple(
    distribution_store: ClassDistributionStore, id_to_probs: Mapping[IdentifierType, List[float]]
):
    probs_data = {k: np.array(v) for k, v in id_to_probs.items()}
    distribution_store.set_probs_multiple(id_to_probs=probs_data)
    assert distribution_store.n_items == len(probs_data)


@pytest.mark.unit
def test_get_logits_raises_on_unknown_id(distribution_store: ClassDistributionStore):
    with pytest.raises(KeyError, match="Unknown identifier"):
        distribution_store.get_logits(identifier=999)


@pytest.mark.unit
def test_set_concentrated_raises_on_unknown_class(distribution_store: ClassDistributionStore):
    with pytest.raises(ValueError, match="Unknown class"):
        distribution_store.set_concentrated(identifier=0, class_name="UNKNOWN")


@pytest.mark.unit
@pytest.mark.parametrize("invalid_idx", [-1, 3, 100])
def test_set_concentrated_idx_raises_on_invalid_index(
    distribution_store: ClassDistributionStore, invalid_idx: int
):
    with pytest.raises(IndexError, match="Class index out of range"):
        distribution_store.set_concentrated_idx(identifier=0, class_idx=invalid_idx)


@pytest.mark.unit
@pytest.mark.parametrize(
    "invalid_probs, error_match",
    [
        ([0.5, 0.5, 0.5], "must sum to 1"),
        ([-0.1, 0.6, 0.5], "non-negative"),
    ],
)
def test_set_probs_raises_on_invalid(
    distribution_store: ClassDistributionStore, invalid_probs: List[float], error_match: str
):
    with pytest.raises(ValueError, match=error_match):
        distribution_store.set_probs(identifier=0, probs=np.array(invalid_probs))


@pytest.mark.unit
def test_set_logits_raises_on_wrong_shape(distribution_store: ClassDistributionStore):
    wrong_shape = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="must be shape"):
        distribution_store.set_logits(identifier=0, logits=wrong_shape)


@pytest.mark.unit
@pytest.mark.parametrize(
    "identifier, logits",
    [
        (0, [1.0, 2.0, 3.0]),
        (5, [0.0, 0.0, 0.0]),
    ],
)
def test_repr(
    distribution_store: ClassDistributionStore, identifier: IdentifierType, logits: List[float]
):
    distribution_store.set_logits(identifier=identifier, logits=np.array(logits))
    repr_str = repr(distribution_store)
    assert "ClassDistributionStore" in repr_str
    assert distribution_store.label_name in repr_str
    assert "n_items=1" in repr_str
