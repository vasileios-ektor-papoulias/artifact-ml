from typing import Any, Dict, Optional, Sequence, Type, TypeVar

from artifact_core._libs.resource_specs.binary_classification.protocol import (
    BinaryClassSpecProtocol,
)
from artifact_core._libs.resource_specs.classification.spec import ClassSpec
from artifact_core._libs.tools.schema.feature_spec.categorical import CategoricalFeatureSpec

BinaryClassSpecT = TypeVar("BinaryClassSpecT", bound="BinaryClassSpec")


class BinaryClassSpec(ClassSpec, BinaryClassSpecProtocol):
    _positive_class_key = "positive_class"

    def __init__(
        self, class_names: Sequence[str], positive_class: str, label_name: Optional[str] = None
    ):
        ls_class_names = list(class_names)
        self._assert_binary(class_names=ls_class_names)
        self._validate_positive_class(positive_class=positive_class, class_names=ls_class_names)
        super().__init__(class_names=class_names, label_name=label_name)
        self._positive_class = positive_class
        self._negative_class = self._get_negative_class(
            positive_class=positive_class, class_names=class_names
        )

    @property
    def positive_class(self) -> str:
        return self._positive_class

    @property
    def negative_class(self) -> str:
        return self._negative_class

    @property
    def positive_class_idx(self) -> int:
        return self.get_class_idx(class_name=self._positive_class)

    @property
    def negative_class_idx(self) -> int:
        return self.get_class_idx(class_name=self._negative_class)

    def __repr__(self) -> str:
        return (
            "BinaryClassSpec("
            f"label_name={self.label_name!r}, "
            f"classes={self.class_names}, "
            f"positive_class={self._positive_class!r}, "
            f"negative_class={self._negative_class!r})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BinaryClassSpec):
            return NotImplemented
        return super().__eq__(other) and self._positive_class == other._positive_class

    def is_positive(self, class_name: str) -> bool:
        if not self.has_class(class_name):
            raise ValueError(f"Unknown class '{class_name}'. Known classes: {self.class_names}")
        return class_name == self._positive_class

    def is_negative(self, class_name: str) -> bool:
        if not self.has_class(class_name):
            raise ValueError(f"Unknown class '{class_name}'. Known classes: {self.class_names}")
        return class_name == self._negative_class

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base[self._positive_class_key] = self._positive_class
        return base

    @classmethod
    def from_dict(cls: Type[BinaryClassSpecT], data: Dict[str, Any]) -> BinaryClassSpecT:
        label_name = cls._get_from_data(key=cls._label_name_key, data=data)
        positive_class = cls._get_from_data(key=cls._positive_class_key, data=data)
        categorical_spec_data = cls._get_from_data(key=cls._categorical_spec_key, data=data)
        categorical_spec = CategoricalFeatureSpec.from_dict(categorical_spec_data)
        return cls(
            label_name=label_name,
            class_names=categorical_spec.ls_categories,
            positive_class=positive_class,
        )

    @staticmethod
    def _get_negative_class(positive_class: str, class_names: Sequence[str]) -> str:
        negative_class = class_names[0] if class_names[1] == positive_class else class_names[1]
        return negative_class

    @staticmethod
    def _assert_binary(class_names: Sequence[str]) -> None:
        if len(class_names) != 2:
            raise ValueError(
                f"`class_names` must contain exactly 2 categories; got {len(class_names)}."
            )
        if len(set(class_names)) != 2:
            raise ValueError(
                f"`class_names` must contain two distinct categories; got {class_names}."
            )

    @staticmethod
    def _validate_positive_class(positive_class: str, class_names: Sequence[str]):
        if positive_class not in class_names:
            raise ValueError(f"`positive_class` {positive_class!r} not in {class_names=}.")
