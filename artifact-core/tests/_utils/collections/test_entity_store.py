from typing import Dict, List, Optional, Tuple

import pytest
from artifact_core._utils.collections.entity_store import EntityStore


@pytest.mark.unit
@pytest.mark.parametrize(
    "initial_data, expected_len",
    [
        (None, 0),
        ({}, 0),
        ({"id1": "value1", "id2": "value2"}, 2),
        ({"a": 1, "b": 2, "c": 3}, 3),
        ({1: "one", 2: "two"}, 2),
    ],
)
def test_init(initial_data: Optional[Dict], expected_len: int):
    store = EntityStore(initial=initial_data)
    assert store.n_items == expected_len
    assert len(store) == expected_len


@pytest.mark.unit
@pytest.mark.parametrize(
    "initial_data, identifier, expected_value",
    [
        ({"id1": "value1", "id2": "value2"}, "id1", "value1"),
        ({"a": 1, "b": 2}, "b", 2),
        ({1: "one", 2: "two"}, 1, "one"),
        ({"key": None}, "key", None),
    ],
)
def test_get(initial_data: Dict, identifier, expected_value):
    store = EntityStore(initial=initial_data)
    result = store.get(identifier)
    assert result == expected_value


@pytest.mark.unit
@pytest.mark.parametrize(
    "initial_data, identifier, value, expected_len",
    [
        ({}, "id1", "value1", 1),
        ({"id1": "old"}, "id1", "new", 1),  # Update existing
        ({"id1": "v1"}, "id2", "v2", 2),  # Add new
        ({}, 1, "one", 1),
        ({}, (1, 2), "tuple_key", 1),
    ],
)
def test_set(initial_data: Dict, identifier, value, expected_len: int):
    store = EntityStore(initial=initial_data)
    store.set(identifier=identifier, value=value)

    assert identifier in store
    assert store.get(identifier) == value
    assert store.n_items == expected_len


@pytest.mark.unit
@pytest.mark.parametrize(
    "initial_data, mapping, expected_total",
    [
        ({}, {"id1": "v1", "id2": "v2"}, 2),
        ({"id1": "v1"}, {"id2": "v2", "id3": "v3"}, 3),
        ({"id1": "old"}, {"id1": "new", "id2": "v2"}, 2),
    ],
)
def test_set_multiple(initial_data: Dict, mapping: Dict, expected_total: int):
    store = EntityStore(initial=initial_data)
    store.set_multiple(mapping=mapping)
    assert store.n_items == expected_total
    for identifier, value in mapping.items():
        assert store.get(identifier) == value


@pytest.mark.unit
@pytest.mark.parametrize(
    "initial_data, identifier_to_delete, expected_remaining",
    [
        ({"id1": "v1", "id2": "v2"}, "id1", 1),
        ({"id1": "v1", "id2": "v2", "id3": "v3"}, "id2", 2),
        ({"only": "value"}, "only", 0),
    ],
)
def test_delete(initial_data: Dict, identifier_to_delete, expected_remaining: int):
    store = EntityStore(initial=initial_data)
    store.delete(identifier=identifier_to_delete)
    assert store.n_items == expected_remaining
    assert identifier_to_delete not in store


@pytest.mark.unit
@pytest.mark.parametrize(
    "initial_data",
    [
        {"id1": "v1", "id2": "v2", "id3": "v3"},
        {"a": 1},
        {},
    ],
)
def test_clear(initial_data: Dict):
    store = EntityStore(initial=initial_data)
    store.clear()
    assert store.n_items == 0
    assert len(store) == 0


@pytest.mark.unit
@pytest.mark.parametrize(
    "initial_data, expected_keys, expected_values, expected_items",
    [
        (
            {"id1": "value1", "id2": "value2"},
            ["id1", "id2"],
            ["value1", "value2"],
            [("id1", "value1"), ("id2", "value2")],
        ),
        (
            {"a": 1},
            ["a"],
            [1],
            [("a", 1)],
        ),
        (
            {},
            [],
            [],
            [],
        ),
    ],
)
def test_properties(
    initial_data: Dict,
    expected_keys: List,
    expected_values: List,
    expected_items: List[Tuple],
):
    store = EntityStore(initial=initial_data)
    assert list(store.keys) == expected_keys
    assert list(store.values) == expected_values
    assert list(store.items) == expected_items
    assert list(store.ids) == expected_keys
    assert store.to_dict == initial_data
    assert len(store) == len(initial_data)


@pytest.mark.unit
@pytest.mark.parametrize(
    "initial_data, identifier_to_check, should_contain",
    [
        ({"id1": "v1", "id2": "v2"}, "id1", True),
        ({"id1": "v1", "id2": "v2"}, "id3", False),
        ({}, "anything", False),
    ],
)
def test_contains(initial_data: Dict, identifier_to_check, should_contain: bool):
    store = EntityStore(initial=initial_data)
    assert (identifier_to_check in store) == should_contain


@pytest.mark.unit
@pytest.mark.parametrize(
    "initial_data, expected_identifiers",
    [
        ({"id1": "v1", "id2": "v2"}, ["id1", "id2"]),
        ({"a": 1}, ["a"]),
        ({}, []),
    ],
)
def test_iter(initial_data: Dict, expected_identifiers: List):
    store = EntityStore(initial=initial_data)
    identifiers = list(store)
    assert identifiers == expected_identifiers


@pytest.mark.unit
def test_get_raises():
    store = EntityStore(initial={"known": "value"})
    with pytest.raises(KeyError, match="Unknown identifier: 'unknown'"):
        store.get(identifier="unknown")


@pytest.mark.unit
def test_delete_raises():
    store = EntityStore(initial={"known": "value"})
    with pytest.raises(KeyError, match="Unknown identifier: 'unknown'"):
        store.delete(identifier="unknown")
