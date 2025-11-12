from typing import Any, Dict, Type, TypeVar

from artifact_core._libs.resource_specs.interfaces.serializable import Serializable
from artifact_core._libs.resource_specs.table_comparison.types import (
    TABULAR_DATA_TYPE_REGISTRY,
    TabularDataDType,
    TabularDataTypeSerializer,
)

ColumnSpecT = TypeVar("ColumnSpecT", bound="ColumnSpec")


class ColumnSpec(Serializable):
    _type_registry = TABULAR_DATA_TYPE_REGISTRY
    _dtype_key = "dtype"

    def __init__(self, dtype: TabularDataDType):
        self._dtype = dtype

    @property
    def dtype(self) -> TabularDataDType:
        return self._dtype

    @classmethod
    def from_dict(cls: Type[ColumnSpecT], data: Dict[str, Any]) -> ColumnSpecT:
        dtype_name = cls._get_from_data(key=cls._dtype_key, data=data)
        dtype = TabularDataTypeSerializer.get_dtype(dtype_name=dtype_name)
        return cls(dtype=dtype)

    def to_dict(self) -> Dict[str, Any]:
        dtype_name = TabularDataTypeSerializer.get_dtype_name(dtype=self._dtype)
        return {self._dtype_key: dtype_name}
