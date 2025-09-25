from typing import Any, Dict, List, Type, TypeVar

from artifact_core.libs.resource_spec.tabular.column_spec import ColumnSpec
from artifact_core.libs.resource_spec.tabular.types import TabularDataTypeSerializer

CategoricalColumnSpecT = TypeVar("CategoricalColumnSpecT", bound="CategoricalColumnSpec")


class CategoricalColumnSpec(ColumnSpec):
    _categories_key = "ls_categories"

    def __init__(self, dtype, ls_categories: List[str]):
        super().__init__(dtype=dtype)
        self._validate_categories(ls_categories=ls_categories)
        self._ls_categories = list(ls_categories)

    @property
    def ls_categories(self) -> List[str]:
        return self._ls_categories.copy()

    @property
    def n_categories(self) -> int:
        return len(self._ls_categories)

    @classmethod
    def from_dict(
        cls: Type[CategoricalColumnSpecT], data: Dict[str, Any]
    ) -> CategoricalColumnSpecT:
        dtype_name = cls._get_from_data(key=cls._dtype_key, data=data)
        dtype = TabularDataTypeSerializer.get_dtype(dtype_name=dtype_name)
        ls_categories = cls._get_from_data(key=cls._categories_key, data=data)
        cls._validate_categories(ls_categories)
        return cls(dtype=dtype, ls_categories=ls_categories)

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base[self._categories_key] = self._ls_categories.copy()
        return base

    @staticmethod
    def _validate_categories(ls_categories: List[str]) -> None:
        if not isinstance(ls_categories, list):
            raise TypeError("`ls_categories` must be a list of strings.")
        if not all(isinstance(c, str) for c in ls_categories):
            raise TypeError("All entries in `ls_categories` must be strings.")
        if len(set(ls_categories)) != len(ls_categories):
            raise ValueError(f"`ls_categories` must not contain duplicates: {ls_categories}")
