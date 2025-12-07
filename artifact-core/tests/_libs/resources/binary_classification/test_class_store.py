from typing import List, Mapping

import pytest
from artifact_core._libs.resource_specs.binary_classification.spec import (
    BinaryClassSpec,
)
from artifact_core._libs.resources.binary_classification.class_store import (
    BinaryClassStore,
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
    store = BinaryClassStore.build_empty(class_spec=spec)
    assert store.class_spec == spec
    assert store.n_items == 0
    assert store.label_name == label_name
    assert list(store.class_names) == class_names
    assert store.n_classes == 2


@pytest.mark.unit
@pytest.mark.parametrize(
    "id_to_class_idx",
    [
        {0: 0, 1: 1},
        {0: 1, 1: 0},
        {0: 0, 1: 1, 2: 0, 3: 1},
    ],
)
def test_from_class_indices(
    binary_spec_params: BinarySpecParams,
    id_to_class_idx: Mapping[IdentifierType, int],
):
    store = BinaryClassStore.from_class_indices(
        class_names=list(binary_spec_params.class_names),
        positive_class=binary_spec_params.positive_class,
        label_name=binary_spec_params.label_name,
        id_to_class_idx=id_to_class_idx,
    )
    assert store.n_items == len(id_to_class_idx)
    for identifier, class_idx in id_to_class_idx.items():
        expected_class = binary_spec_params.class_names[class_idx]
        assert store.get_class_name(identifier=identifier) == expected_class


@pytest.mark.unit
@pytest.mark.parametrize(
    "id_to_class_idx",
    [
        {0: 0, 1: 1},
        {0: 1, 1: 0},
        {0: 0, 1: 1, 2: 0, 3: 1},
    ],
)
def test_from_class_indices_and_spec(
    binary_spec: BinaryClassSpec,
    id_to_class_idx: Mapping[IdentifierType, int],
):
    store = BinaryClassStore.from_class_indices_and_spec(
        class_spec=binary_spec, id_to_class_idx=id_to_class_idx
    )
    assert store.n_items == len(id_to_class_idx)
    assert store.class_spec == binary_spec
    for identifier, class_idx in id_to_class_idx.items():
        assert store.get_class_idx(identifier=identifier) == class_idx


@pytest.mark.unit
@pytest.mark.parametrize(
    "id_to_class",
    [
        {0: "neg", 1: "pos"},
        {0: "pos", 1: "neg"},
        {0: "neg", 1: "pos", 2: "pos", 3: "neg"},
    ],
)
def test_from_class_names(
    binary_spec_params: BinarySpecParams,
    id_to_class: Mapping[IdentifierType, str],
):
    store = BinaryClassStore.from_class_names(
        class_names=list(binary_spec_params.class_names),
        positive_class=binary_spec_params.positive_class,
        id_to_class=id_to_class,
    )
    assert store.n_items == len(id_to_class)
    for identifier, expected_class in id_to_class.items():
        assert store.get_class_name(identifier=identifier) == expected_class


@pytest.mark.unit
@pytest.mark.parametrize(
    "id_to_class",
    [
        {0: "neg", 1: "pos"},
        {0: "pos", 1: "neg"},
        {0: "neg", 1: "pos", 2: "pos", 3: "neg"},
    ],
)
def test_from_class_names_and_spec(
    binary_spec: BinaryClassSpec,
    id_to_class: Mapping[IdentifierType, str],
):
    store = BinaryClassStore.from_class_names_and_spec(
        class_spec=binary_spec, id_to_class=id_to_class
    )
    assert store.n_items == len(id_to_class)
    assert store.id_to_class_name == id_to_class


@pytest.mark.unit
@pytest.mark.parametrize(
    "identifier, class_name, expected_is_positive",
    [
        (0, "pos", True),
        (1, "neg", False),
        (2, "pos", True),
    ],
)
def test_id_to_is_positive_and_negative(
    binary_class_store: BinaryClassStore,
    identifier: IdentifierType,
    class_name: str,
    expected_is_positive: bool,
):
    binary_class_store.set_class(identifier=identifier, class_name=class_name)
    assert binary_class_store.id_to_is_positive[identifier] is expected_is_positive
    assert binary_class_store.id_to_is_negative[identifier] is (not expected_is_positive)


@pytest.mark.unit
@pytest.mark.parametrize(
    "class_name, expected_positive, expected_negative",
    [
        ("neg", False, True),
        ("pos", True, False),
    ],
)
def test_stored_class_positivity(
    binary_class_store: BinaryClassStore,
    class_name: str,
    expected_positive: bool,
    expected_negative: bool,
):
    binary_class_store.set_class(identifier=0, class_name=class_name)
    assert binary_class_store.stored_class_is_positive(identifier=0) is expected_positive
    assert binary_class_store.stored_class_is_negative(identifier=0) is expected_negative


@pytest.mark.unit
@pytest.mark.parametrize(
    "id_to_class",
    [
        {0: "neg", 1: "pos"},
        {0: "pos", 1: "neg"},
    ],
)
def test_inherits_class_store_methods(
    binary_class_store: BinaryClassStore,
    binary_spec_params: BinarySpecParams,
    id_to_class: Mapping[IdentifierType, str],
):
    binary_class_store.set_multiple_cat(id_to_class=id_to_class)
    first_id = next(iter(id_to_class.keys()))
    first_class = id_to_class[first_id]
    neg_class = binary_spec_params.negative_class
    expected_idx = 0 if first_class == neg_class else 1

    assert binary_class_store.get_class_idx(identifier=first_id) == expected_idx
    assert binary_class_store.get_class_name(identifier=first_id) == first_class
    assert binary_class_store.n_classes == 2
