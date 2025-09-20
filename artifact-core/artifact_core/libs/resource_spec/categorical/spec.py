from typing import Dict, List, TypeVar

from artifact_core.libs.resource_spec.categorical.protocol import (
    CategoricalFeatureSpecProtocol,
)

CategoricalFeatureSpecT = TypeVar("CategoricalFeatureSpecT", bound="CategoricalFeatureSpec")


class CategoricalFeatureSpec(CategoricalFeatureSpecProtocol):
    def __init__(self, feature_name: str, ls_categories: List[str]):
        self._validate_ls_categories(ls_categories)
        self._feature_name: str = str(feature_name)
        self._ls_categories: List[str] = ls_categories.copy()
        self._cat_to_idx: Dict[str, int] = {c: i for i, c in enumerate(self._ls_categories)}

    @property
    def feature_name(self) -> str:
        return self._feature_name

    @property
    def ls_categories(self) -> List[str]:
        return self._ls_categories.copy()

    @property
    def n_categories(self) -> int:
        return len(self._ls_categories)

    def __repr__(self) -> str:
        return (
            f"CategoricalFeature(feature_name={self._feature_name!r}, "
            f"n_categories={self.n_categories}, "
            f"categories={self._ls_categories})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CategoricalFeatureSpec):
            return NotImplemented
        return (
            self._feature_name == other._feature_name
            and self._ls_categories == other._ls_categories
        )

    def get_category_idx(self, category: str) -> int:
        self._require_category(category)
        return self._cat_to_idx[category]

    def has_category(self, category: str) -> bool:
        return category in self._cat_to_idx

    def _require_category(self, category: str) -> None:
        if category not in self._cat_to_idx:
            raise ValueError(
                f"Unknown category '{category}'. Known categories: {self._ls_categories}"
            )

    @staticmethod
    def _validate_ls_categories(ls_categories: List[str]) -> None:
        if not ls_categories:
            raise ValueError("`ls_categories` must be a non-empty list of strings.")
        if not all(isinstance(c, str) for c in ls_categories):
            raise TypeError("All categories must be strings.")
