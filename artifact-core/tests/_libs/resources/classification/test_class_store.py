from typing import List, Mapping

import pytest
from artifact_core._libs.resource_specs.classification.spec import ClassSpec
from artifact_core._libs.resources.classification.class_store import ClassStore
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
    store = ClassStore(class_spec=spec)
    assert store.class_spec == spec
    assert store.n_items == 0
    assert store.label_name == label_name
    assert list(store.class_names) == class_names
    assert store.n_classes == len(class_names)


@pytest.mark.unit
@pytest.mark.parametrize(
    "class_names, label_name, id_to_class_idx",
    [
        (["A", "B", "C"], "target", {0: 0, 1: 1, 2: 2}),
        (["neg", "pos"], "label", {0: 0, 1: 1}),
        (["A", "B", "C"], "target", {0: 0, 1: 1, 2: 2, 3: 0}),
        (["cat", "dog"], "animal", {10: 0, 20: 1, 30: 0}),
    ],
)
def test_init_with_data(
    class_names: List[str], label_name: str, id_to_class_idx: Mapping[IdentifierType, int]
):
    spec = ClassSpec(class_names=class_names, label_name=label_name)
    store = ClassStore(class_spec=spec, id_to_class_idx=id_to_class_idx)
    assert store.n_items == len(id_to_class_idx)
    assert store.id_to_class_idx == id_to_class_idx


@pytest.mark.unit
@pytest.mark.parametrize(
    "identifier, class_idx, expected_class_name",
    [(0, 0, "A"), (1, 1, "B"), (2, 2, "C"), (99, 0, "A")],
)
def test_set_class_idx(
    class_spec: ClassSpec, identifier: IdentifierType, class_idx: int, expected_class_name: str
):
    store = ClassStore(class_spec=class_spec)
    store.set_class_idx(identifier=identifier, class_idx=class_idx)
    assert store.get_class_idx(identifier=identifier) == class_idx
    assert store.get_class_name(identifier=identifier) == expected_class_name


@pytest.mark.unit
@pytest.mark.parametrize(
    "identifier, class_name, expected_class_idx",
    [(0, "A", 0), (1, "B", 1), (2, "C", 2), (99, "A", 0)],
)
def test_set_class(
    class_spec: ClassSpec, identifier: IdentifierType, class_name: str, expected_class_idx: int
):
    store = ClassStore(class_spec=class_spec)
    store.set_class(identifier=identifier, class_name=class_name)
    assert store.get_class_idx(identifier=identifier) == expected_class_idx
    assert store.get_class_name(identifier=identifier) == class_name


@pytest.mark.unit
@pytest.mark.parametrize(
    "id_to_class_idx",
    [{0: 0, 1: 1, 2: 2}, {0: 0, 1: 1, 2: 2, 3: 0}, {10: 1, 20: 2}],
)
def test_set_multiple_idx(class_spec: ClassSpec, id_to_class_idx: Mapping[IdentifierType, int]):
    store = ClassStore(class_spec=class_spec)
    store.set_multiple_idx(id_to_class_idx=id_to_class_idx)
    assert store.n_items == len(id_to_class_idx)
    assert store.id_to_class_idx == id_to_class_idx


@pytest.mark.unit
@pytest.mark.parametrize(
    "id_to_class",
    [{0: "A", 1: "B", 2: "C"}, {0: "A", 1: "B", 2: "C", 3: "A"}, {10: "B", 20: "C"}],
)
def test_set_multiple_cat(class_spec: ClassSpec, id_to_class: Mapping[IdentifierType, str]):
    store = ClassStore(class_spec=class_spec)
    store.set_multiple_cat(id_to_class=id_to_class)
    assert store.n_items == len(id_to_class)
    assert store.id_to_class_name == id_to_class


@pytest.mark.unit
@pytest.mark.parametrize(
    "id_to_class_idx, expected_class_names",
    [
        ({0: 0, 1: 1, 2: 2}, ["A", "B", "C"]),
        ({0: 0, 1: 0, 2: 0}, ["A", "A", "A"]),
        ({0: 2, 1: 1}, ["C", "B"]),
    ],
)
def test_stored_indices_and_names(
    class_spec: ClassSpec,
    id_to_class_idx: Mapping[IdentifierType, int],
    expected_class_names: List[str],
):
    store = ClassStore(class_spec=class_spec, id_to_class_idx=id_to_class_idx)
    assert list(store.stored_indices) == list(id_to_class_idx.values())
    assert list(store.stored_class_names) == expected_class_names


@pytest.mark.unit
def test_get_class_idx_raises_on_unknown_id(class_spec: ClassSpec):
    store = ClassStore(class_spec=class_spec)
    with pytest.raises(KeyError, match="Unknown identifier"):
        store.get_class_idx(identifier=999)


@pytest.mark.unit
def test_set_class_raises_on_unknown_class(class_spec: ClassSpec):
    store = ClassStore(class_spec=class_spec)
    with pytest.raises(ValueError, match="Unknown class"):
        store.set_class(identifier=0, class_name="UNKNOWN")


@pytest.mark.unit
@pytest.mark.parametrize("invalid_idx", [-1, 3, 100])
def test_set_class_idx_raises_on_invalid_index(class_spec: ClassSpec, invalid_idx: int):
    store = ClassStore(class_spec=class_spec)
    with pytest.raises(IndexError, match="Class index out of range"):
        store.set_class_idx(identifier=0, class_idx=invalid_idx)


@pytest.mark.unit
def test_set_class_idx_raises_on_non_int(class_spec: ClassSpec):
    store = ClassStore(class_spec=class_spec)
    with pytest.raises(TypeError, match="must be an integer"):
        store.set_class_idx(identifier=0, class_idx="not_int")  # type: ignore


@pytest.mark.unit
@pytest.mark.parametrize(
    "id_to_class_idx",
    [{0: 0, 1: 1, 2: 2}, {0: 0, 1: 1, 2: 2, 3: 0}],
)
def test_repr(class_spec: ClassSpec, id_to_class_idx: Mapping[IdentifierType, int]):
    store = ClassStore(class_spec=class_spec, id_to_class_idx=id_to_class_idx)
    repr_str = repr(store)
    assert "ClassStore" in repr_str
    assert class_spec.label_name in repr_str
    assert f"n_items={len(id_to_class_idx)}" in repr_str
