from typing import Any, Dict, List, Sequence, Type, TypeVar

from artifact_core._libs.tools.schema.data_types.serializer import TabularDataTypeSerializer
from artifact_core._libs.tools.schema.data_types.typing import TabularDataDType
from artifact_core._libs.tools.schema.feature_spec.feature_spec import FeatureSpec

CategoricalFeatureSpecT = TypeVar("CategoricalFeatureSpecT", bound="CategoricalFeatureSpec")


class CategoricalFeatureSpec(FeatureSpec):
    _ls_categories_key = "ls_categories"

    def __init__(self, dtype: TabularDataDType, ls_categories: List[str]):
        super().__init__(dtype=dtype)
        self._validate_categories(ls_categories=ls_categories)
        self._ls_categories = ls_categories

    @property
    def ls_categories(self) -> List[str]:
        return self._ls_categories.copy()

    @property
    def n_categories(self) -> int:
        return len(self._ls_categories)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CategoricalFeatureSpec):
            return NotImplemented
        return super().__eq__(other) and self._ls_categories == other._ls_categories

    @classmethod
    def from_dict(
        cls: Type[CategoricalFeatureSpecT], data: Dict[str, Any]
    ) -> CategoricalFeatureSpecT:
        dtype_name = cls._get_from_data(key=cls._dtype_key, data=data)
        dtype = TabularDataTypeSerializer.get_dtype(dtype_name=dtype_name)
        ls_categories = cls._get_from_data(key=cls._ls_categories_key, data=data)
        cls._validate_categories(ls_categories)
        return cls(dtype=dtype, ls_categories=ls_categories)

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base[self._ls_categories_key] = self._ls_categories.copy()
        return base

    @staticmethod
    def _validate_categories(ls_categories: Sequence[str]) -> None:
        if not isinstance(ls_categories, list):
            raise TypeError("ls_categories must be a list of strings.")
        if not all(isinstance(c, str) for c in ls_categories):
            raise TypeError("All entries in `categories` must be strings.")
        if len(set(ls_categories)) != len(ls_categories):
            raise ValueError(f"`categories` must not contain duplicates: {ls_categories}")
