from typing import List

from artifact_core.libs.resource_spec.categorical.spec import CategoricalFeatureSpec


class BinaryFeatureSpec(CategoricalFeatureSpec):
    def __init__(self, feature_name: str, positive_category: str, ls_categories: List[str]):
        self._assert_binary(ls_categories=ls_categories)
        self._validate_positive_category(
            positive_category=positive_category, ls_categories=ls_categories
        )
        super().__init__(feature_name=feature_name, ls_categories=ls_categories)
        self._positive_category = positive_category
        self._negative_category = self._get_negative_category(
            positive_category=positive_category, ls_categories=ls_categories
        )

    @property
    def positive_category(self) -> str:
        return self._positive_category

    @property
    def negative_category(self) -> str:
        return self._negative_category

    def __repr__(self) -> str:
        return (
            "BinaryFeatureSpec("
            f"feature_name={self.feature_name!r}, "
            f"categories={self.ls_categories}, "
            f"positive_category={self._positive_category!r}, "
            f"negative_category={self._negative_category!r})"
        )

    def is_positive(self, category: str) -> bool:
        self._require_category(category)
        return category == self._positive_category

    def is_negative(self, category: str) -> bool:
        self._require_category(category)
        return category == self._negative_category

    @staticmethod
    def _get_negative_category(positive_category: str, ls_categories: List[str]) -> str:
        negative_category = (
            ls_categories[0] if ls_categories[1] == positive_category else ls_categories[1]
        )
        return negative_category

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
    def _validate_positive_category(positive_category: str, ls_categories: List[str]):
        if positive_category not in ls_categories:
            raise ValueError(
                f"`positive_category` {positive_category!r} must be one of ls_categories={ls_categories}."
            )
