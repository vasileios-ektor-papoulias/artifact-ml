import json
from abc import ABC, abstractmethod
from typing import Any, Dict, Type, TypeVar

SerializableT = TypeVar("SerializableT", bound="Serializable")


class Serializable(ABC):
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]: ...

    @classmethod
    @abstractmethod
    def from_dict(cls: Type[SerializableT], data: Dict[str, Any]) -> SerializableT: ...

    def serialize(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def deserialize(cls: Type[SerializableT], json_str: str) -> SerializableT:
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON object: {e}") from e
        if not isinstance(data, dict):
            raise ValueError("Expected JSON object.")
        return cls.from_dict(data=data)

    @staticmethod
    def _get_from_data(key: str, data: Dict[str, Any]) -> Any:
        value = data.get(key)
        if value is None:
            raise KeyError(f"required key: '{key}', got keys: {data.keys()}")
        return value
