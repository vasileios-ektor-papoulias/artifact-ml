from typing import List

from artifact_core.libs.resource_spec.categorical.spec import CategoricalFeatureSpec


class BinaryFeatureSpec(CategoricalFeatureSpec):
    def __init__(self, feature_name: str, true_category: str, ls_categories: List[str]):
        self._assert_binary(ls_categories=ls_categories)
        self._validate_true_category(true_category=true_category, ls_categories=ls_categories)
        super().__init__(feature_name=feature_name, ls_categories=ls_categories)
        self._true_category = true_category
        self._false_category = self._get_false_category(
            true_category=true_category, ls_categories=ls_categories
        )

    @property
    def true_category(self) -> str:
        return self._true_category

    @property
    def false_category(self) -> str:
        return self._false_category

    def __repr__(self) -> str:
        return (
            "BinaryFeatureSpec("
            f"feature_name={self.feature_name!r}, "
            f"categories={self.ls_categories}, "
            f"true_category={self._true_category!r}, "
            f"false_category={self._false_category!r})"
        )

    def is_positive(self, category: str) -> bool:
        self._require_category(category)
        return category == self._true_category

    def is_negative(self, category: str) -> bool:
        self._require_category(category)
        return category == self._false_category

    @staticmethod
    def _get_false_category(true_category: str, ls_categories: List[str]) -> str:
        false_category = ls_categories[0] if ls_categories[1] == true_category else ls_categories[1]
        return false_category

    @staticmethod
    def _assert_binary(ls_categories: List[str]) -> None:
        if len(ls_categories) != 2:
            raise ValueError(
                f"`ls_categories` must contain exactly 2 categories; got {len(ls_categories)}."
            )
        if len(set(ls_categories)) != 2:
            raise ValueError(
                f"`ls_categories` must contain two DISTINCT categories; got {ls_categories}."
            )

    @staticmethod
    def _validate_true_category(true_category: str, ls_categories: List[str]):
        if true_category not in ls_categories:
            raise ValueError(
                f"`true_category` {true_category!r} must be one of ls_categories={ls_categories}."
            )
