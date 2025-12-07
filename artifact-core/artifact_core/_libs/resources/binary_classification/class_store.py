from typing import Dict, Mapping, Optional, Sequence, Type, TypeVar

from artifact_core._libs.resource_specs.binary_classification.protocol import (
    BinaryClassSpecProtocol,
)
from artifact_core._libs.resource_specs.binary_classification.spec import BinaryClassSpec
from artifact_core._libs.resources.classification.class_store import ClassStore
from artifact_core._utils.collections.entity_store import IdentifierType

BinaryClassStoreT = TypeVar("BinaryClassStoreT", bound="BinaryClassStore")


class BinaryClassStore(ClassStore[BinaryClassSpecProtocol]):
    @classmethod
    def from_class_indices(
        cls: Type[BinaryClassStoreT],
        class_names: Sequence[str],
        positive_class: str,
        label_name: Optional[str] = None,
        id_to_class_idx: Optional[Mapping[IdentifierType, int]] = None,
    ) -> BinaryClassStoreT:
        class_spec = BinaryClassSpec(
            class_names=class_names, positive_class=positive_class, label_name=label_name
        )
        store = cls.from_class_indices_and_spec(
            class_spec=class_spec, id_to_class_idx=id_to_class_idx
        )
        return store

    @classmethod
    def from_class_indices_and_spec(
        cls: Type[BinaryClassStoreT],
        class_spec: BinaryClassSpecProtocol,
        id_to_class_idx: Optional[Mapping[IdentifierType, int]] = None,
    ) -> BinaryClassStoreT:
        store = cls(class_spec=class_spec, id_to_class_idx=id_to_class_idx)
        return store

    @classmethod
    def from_class_names(
        cls: Type[BinaryClassStoreT],
        class_names: Sequence[str],
        positive_class: str,
        label_name: Optional[str] = None,
        id_to_class: Optional[Mapping[IdentifierType, str]] = None,
    ) -> BinaryClassStoreT:
        class_spec = BinaryClassSpec(
            class_names=class_names, positive_class=positive_class, label_name=label_name
        )
        store = cls.from_class_names_and_spec(class_spec=class_spec, id_to_class=id_to_class)
        return store

    @classmethod
    def from_class_names_and_spec(
        cls: Type[BinaryClassStoreT],
        class_spec: BinaryClassSpecProtocol,
        id_to_class: Optional[Mapping[IdentifierType, str]] = None,
    ) -> BinaryClassStoreT:
        if id_to_class is None:
            id_to_class = {}
        store = cls(class_spec=class_spec, id_to_class_idx=None)
        store.set_multiple_cat(id_to_class=id_to_class)
        return store

    @property
    def id_to_is_positive(self) -> Dict[IdentifierType, bool]:
        return {
            identifier: self.stored_class_is_positive(identifier=identifier)
            for identifier in self._data.keys()
        }

    @property
    def id_to_is_negative(self) -> Dict[IdentifierType, bool]:
        return {
            identifier: self.stored_class_is_negative(identifier=identifier)
            for identifier in self._data.keys()
        }

    def stored_class_is_positive(self, identifier: IdentifierType) -> bool:
        class_name = self.get_class_name(identifier=identifier)
        return self._class_spec.is_positive(class_name=class_name)

    def stored_class_is_negative(self, identifier: IdentifierType) -> bool:
        class_name = self.get_class_name(identifier=identifier)
        return self._class_spec.is_negative(class_name=class_name)
