from typing import Generic, Mapping, Optional, Sequence, Type, TypeVar

from artifact_core._libs.resource_specs.classification.protocol import ClassSpecProtocol
from artifact_core._utils.collections.entity_store import EntityStore, IdentifierType

ClassSpecProtocolT = TypeVar("ClassSpecProtocolT", bound=ClassSpecProtocol)
ClassStoreT = TypeVar("ClassStoreT", bound="ClassStore")


class ClassStore(EntityStore[int], Generic[ClassSpecProtocolT]):
    def __init__(
        self,
        class_spec: ClassSpecProtocolT,
        id_to_class_idx: Optional[Mapping[IdentifierType, int]] = None,
    ):
        super().__init__(initial=None)
        self._class_spec = class_spec
        if id_to_class_idx is not None:
            self.set_multiple_idx(id_to_class_idx=id_to_class_idx)

    @classmethod
    def build_empty(cls: Type[ClassStoreT], class_spec: ClassSpecProtocolT) -> ClassStoreT:
        return cls(class_spec=class_spec)

    def __repr__(self) -> str:
        return (
            f"ClassStore(label_name={self._class_spec.label_name!r}, "
            f"n_items={self.n_items}, n_classes={self._class_spec.n_classes})"
        )

    @property
    def class_spec(self) -> ClassSpecProtocolT:
        return self._class_spec

    @property
    def label_name(self) -> str:
        return self._class_spec.label_name

    @property
    def class_names(self) -> Sequence[str]:
        return self._class_spec.class_names

    @property
    def n_classes(self) -> int:
        return self._class_spec.n_classes

    @property
    def id_to_class_idx(self) -> Mapping[IdentifierType, int]:
        return dict(self._data)

    @property
    def id_to_class_name(self) -> Mapping[IdentifierType, str]:
        return {k: self._class_spec.class_names[v] for k, v in self._data.items()}

    @property
    def stored_indices(self) -> Sequence[int]:
        return list(self._data.values())

    @property
    def stored_class_names(self) -> Sequence[str]:
        cats = self._class_spec.class_names
        return [cats[i] for i in self.stored_indices]

    def get_class_idx(self, identifier: IdentifierType) -> int:
        self._require_id(identifier=identifier)
        class_idx = self.get(identifier=identifier)
        return class_idx

    def get_class_name(self, identifier: IdentifierType) -> str:
        class_idx = self.get_class_idx(identifier=identifier)
        class_name = self._class_spec.class_names[class_idx]
        return class_name

    def set_class_idx(self, identifier: IdentifierType, class_idx: int) -> None:
        self._require_class_idx(class_idx=class_idx)
        self.set(identifier=identifier, value=class_idx)

    def set_class(self, identifier: IdentifierType, class_name: str) -> None:
        class_name = self._validate_class(class_name=class_name)
        self._require_class(class_name=class_name)
        class_idx = self._class_spec.get_class_idx(class_name=class_name)
        self.set(identifier=identifier, value=class_idx)

    def set_multiple_idx(self, id_to_class_idx: Mapping[IdentifierType, int]) -> None:
        for identifier, class_idx in id_to_class_idx.items():
            self.set_class_idx(identifier=identifier, class_idx=class_idx)

    def set_multiple_cat(self, id_to_class: Mapping[IdentifierType, str]) -> None:
        for identifier, class_name in id_to_class.items():
            class_name = self._validate_class(class_name=class_name)
            self.set_class(identifier=identifier, class_name=class_name)

    def _require_id(self, identifier: IdentifierType) -> None:
        if identifier not in self._data:
            raise KeyError(
                f"Unknown identifier: {identifier!r}. Known identifiers: {list(self._data.keys())}"
            )

    def _require_class(self, class_name: str) -> None:
        if class_name not in self._class_spec.class_names:
            raise ValueError(
                f"Unknown class '{class_name}'. "
                f"Known classes (in order): {self._class_spec.class_names}"
            )

    def _require_class_idx(self, class_idx: int) -> None:
        if not isinstance(class_idx, int):
            raise TypeError("`class_idx` must be an integer.")
        if not (0 <= class_idx < self._class_spec.n_classes):
            raise IndexError(
                f"Class index out of range: "
                f"0 <= class_idx < {self._class_spec.n_classes}, "
                f"got {class_idx}."
            )

    @staticmethod
    def _validate_class(class_name: str) -> str:
        return str(class_name)
