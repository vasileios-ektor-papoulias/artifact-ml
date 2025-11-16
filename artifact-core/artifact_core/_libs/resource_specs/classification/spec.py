from typing import Any, Dict, Optional, Sequence, Type, TypeVar

from artifact_core._interfaces.serializable import Serializable
from artifact_core._libs.resource_specs.classification.protocol import ClassSpecProtocol
from artifact_core._libs.tools.schema.feature_spec.categorical import CategoricalFeatureSpec

ClassSpecT = TypeVar("ClassSpecT", bound="ClassSpec")


class ClassSpec(Serializable, ClassSpecProtocol):
    _default_label_name = "label"
    _label_name_key = "label_name"
    _categorical_spec_key = "categorical_spec"

    def __init__(
        self,
        class_names: Sequence[str],
        label_name: Optional[str] = None,
    ):
        if label_name is None:
            label_name = self._default_label_name
        self._label_name: str = str(label_name)
        self._categorical_spec = CategoricalFeatureSpec(dtype=str, ls_categories=list(class_names))

    @property
    def label_name(self) -> str:
        return self._label_name

    @property
    def class_names(self) -> Sequence[str]:
        return self._categorical_spec.ls_categories

    @property
    def n_classes(self) -> int:
        return self._categorical_spec.n_categories

    def __repr__(self) -> str:
        return (
            f"ClassSpec(label_name={self._label_name!r}, "
            f"n_classes={self.n_classes}, "
            f"classes={self.class_names})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ClassSpec):
            return NotImplemented
        return (
            self._label_name == other._label_name
            and self._categorical_spec == other._categorical_spec
        )

    def get_class_idx(self, class_name: str) -> int:
        if not self.has_class(class_name):
            raise ValueError(f"Unknown class '{class_name}'. Known classes: {self.class_names}")
        return self._categorical_spec.ls_categories.index(class_name)

    def has_class(self, class_name: str) -> bool:
        return class_name in self._categorical_spec.ls_categories

    def to_dict(self) -> Dict[str, Any]:
        return {
            self._label_name_key: self._label_name,
            self._categorical_spec_key: (self._categorical_spec.to_dict()),
        }

    @classmethod
    def from_dict(cls: Type[ClassSpecT], data: Dict[str, Any]) -> ClassSpecT:
        label_name = cls._get_from_data(key=cls._label_name_key, data=data)
        categorical_spec_data = cls._get_from_data(key=cls._categorical_spec_key, data=data)
        categorical_spec = CategoricalFeatureSpec.from_dict(categorical_spec_data)
        return cls(label_name=label_name, class_names=categorical_spec.ls_categories)
