from typing import Type, TypeVar

from artifact_core._libs.tools.schema.data_types.registry import TABULAR_DATA_TYPE_REGISTRY
from artifact_core._libs.tools.schema.data_types.typing import TabularDataDType

TabularDataTypeSerializerT = TypeVar(
    "TabularDataTypeSerializerT", bound="TabularDataTypeSerializer"
)


class TabularDataTypeSerializer:
    _type_registry = TABULAR_DATA_TYPE_REGISTRY

    @classmethod
    def get_dtype(cls: Type[TabularDataTypeSerializerT], dtype_name: str) -> TabularDataDType:
        if dtype_name not in cls._type_registry:
            raise ValueError(f"Unknown dtype during deserialization: {dtype_name}")
        dtype = cls._type_registry[dtype_name]
        return dtype

    @classmethod
    def get_dtype_name(cls: Type[TabularDataTypeSerializerT], dtype: TabularDataDType) -> str:
        for registered_dtype_name, registered_dtype in cls._type_registry.items():
            if dtype is registered_dtype:
                return registered_dtype_name
        raise ValueError(f"Unsupported dtype for serialization: {dtype}")
