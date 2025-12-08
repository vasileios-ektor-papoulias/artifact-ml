from typing import Any, Callable, Dict, Type, TypeVar

import pytest
from artifact_core._interfaces.serializable import Serializable

DummySerializableT = TypeVar("DummySerializableT", bound="DummySerializable")


class DummySerializable(Serializable):
    def __init__(self, name: str, value: int):
        self.name = name
        self.value = value

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "value": self.value}

    @classmethod
    def from_dict(cls: Type[DummySerializableT], data: Dict[str, Any]) -> DummySerializableT:
        name = cls._get_from_data("name", data)
        value = cls._get_from_data("value", data)
        return cls(name=name, value=value)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DummySerializable):
            return False
        return self.name == other.name and self.value == other.value


@pytest.fixture
def dummy_instance() -> DummySerializable:
    return DummySerializable(name="test", value=42)


@pytest.mark.unit
@pytest.mark.parametrize(
    "method, expected",
    [
        ("to_dict", {"name": "test", "value": 42}),
        ("serialize", '{"name": "test", "value": 42}'),
    ],
)
def test_serialization_methods(dummy_instance: DummySerializable, method: str, expected: Any):
    result = getattr(dummy_instance, method)()
    assert result == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    "factory, input_data",
    [
        (DummySerializable.from_dict, {"name": "test", "value": 42}),
        (DummySerializable.deserialize, '{"name": "test", "value": 42}'),
    ],
)
def test_factory_methods(factory: Callable[..., DummySerializable], input_data: Any):
    result = factory(input_data)
    assert result == DummySerializable(name="test", value=42)


@pytest.mark.unit
@pytest.mark.parametrize(
    "json_str, error_match",
    [
        ("not valid json", "Invalid JSON object"),
        ("[1, 2, 3]", "Expected JSON object"),
    ],
)
def test_deserialize_invalid_input(json_str: str, error_match: str):
    with pytest.raises(ValueError, match=error_match):
        DummySerializable.deserialize(json_str)


@pytest.mark.unit
@pytest.mark.parametrize(
    "name, value",
    [
        ("roundtrip", 123),
        ("another", 0),
        ("negative", -42),
    ],
)
def test_roundtrip(name: str, value: int):
    original = DummySerializable(name=name, value=value)
    json_str = original.serialize()
    restored = DummySerializable.deserialize(json_str)
    assert restored == original


@pytest.mark.unit
@pytest.mark.parametrize(
    "data, key, expected",
    [
        ({"key": "value"}, "key", "value"),
        ({"a": 1, "b": 2}, "a", 1),
        ({"nested": {"inner": "x"}}, "nested", {"inner": "x"}),
    ],
)
def test_get_from_data(data: Dict[str, Any], key: str, expected: Any):
    result = Serializable._get_from_data(key, data)
    assert result == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    "data, key",
    [
        ({"other_key": "value"}, "key"),
        ({"key": None}, "key"),
        ({}, "missing"),
    ],
)
def test_get_from_data_raises_on_missing_or_none(data: Dict[str, Any], key: str):
    with pytest.raises(KeyError, match=f"required key: '{key}'"):
        Serializable._get_from_data(key, data)


@pytest.mark.unit
@pytest.mark.parametrize(
    "data, missing_key",
    [
        ({"name": "test"}, "value"),
        ({"value": 42}, "name"),
        ({}, "name"),
    ],
)
def test_from_dict_raises_on_missing_key(data: Dict[str, Any], missing_key: str):
    with pytest.raises(KeyError, match=f"required key: '{missing_key}'"):
        DummySerializable.from_dict(data)
