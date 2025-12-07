from typing import Any, Dict, Type, TypeVar

from artifact_core._interfaces.serializable import Serializable
from artifact_core._libs.tools.schema.data_types.registry import TABULAR_DATA_TYPE_REGISTRY
from artifact_core._libs.tools.schema.data_types.serializer import TabularDataTypeSerializer
from artifact_core._libs.tools.schema.data_types.typing import TabularDataDType

FeatureSpecT = TypeVar("FeatureSpecT", bound="FeatureSpec")


class FeatureSpec(Serializable):
    _type_registry = TABULAR_DATA_TYPE_REGISTRY
    _dtype_key = "dtype"

    def __init__(self, dtype: TabularDataDType):
        self._dtype = dtype

    @property
    def dtype(self) -> TabularDataDType:
        return self._dtype

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FeatureSpec):
            return NotImplemented
        return self._dtype == other._dtype

    @classmethod
    def from_dict(cls: Type[FeatureSpecT], data: Dict[str, Any]) -> FeatureSpecT:
        dtype_name = cls._get_from_data(key=cls._dtype_key, data=data)
        dtype = TabularDataTypeSerializer.get_dtype(dtype_name=dtype_name)
        return cls(dtype=dtype)

    def to_dict(self) -> Dict[str, Any]:
        dtype_name = TabularDataTypeSerializer.get_dtype_name(dtype=self._dtype)
        return {self._dtype_key: dtype_name}
