from typing import Any, Dict, Type, TypeVar

from artifact_core.libs.resource_spec.tabular.types import (
    TABULAR_DATA_TYPE_REGISTRY,
    TabularDataDType,
    TabularDataTypeSerializer,
)
from artifact_core.libs.types.serializable import Serializable

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
