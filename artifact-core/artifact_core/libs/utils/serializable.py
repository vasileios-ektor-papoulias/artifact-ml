import json
from abc import ABC, abstractmethod
from typing import Any, Dict, Type, TypeVar

SerializableT = TypeVar("SerializableT", bound="Serializable")


class Serializable(ABC):
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]: ...

    @abstractmethod
    @classmethod
    def from_dict(cls: Type[SerializableT], data: Dict[str, Any]) -> SerializableT: ...

    def serialize(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def deserialize(cls: Type[SerializableT], json_str: str) -> SerializableT:
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON for CategoricalFeatureSpec: {e}") from e
        if not isinstance(data, dict):
            raise ValueError("Expected JSON object for CategoricalFeatureSpec.")
        return cls.from_dict(data=data)
