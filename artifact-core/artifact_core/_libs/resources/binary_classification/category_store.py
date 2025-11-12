from typing import Dict, List, Mapping, Optional, Type, TypeVar

from artifact_core._libs.resource_specs.binary_classification.protocol import (
    BinaryFeatureSpecProtocol,
)
from artifact_core._libs.resource_specs.binary_classification.spec import BinaryFeatureSpec
from artifact_core._libs.resources.classification.category_store import CategoryStore
from artifact_core._libs.resources.tools.entity_store import IdentifierType

BinaryCategoryStoreT = TypeVar("BinaryCategoryStoreT", bound="BinaryCategoryStore")


class BinaryCategoryStore(CategoryStore[BinaryFeatureSpecProtocol]):
    @classmethod
    def from_category_indices(
        cls: Type[BinaryCategoryStoreT],
        ls_categories: List[str],
        positive_category: str,
        feature_name: Optional[str] = None,
        id_to_category_idx: Optional[Mapping[IdentifierType, int]] = None,
    ) -> BinaryCategoryStoreT:
        feature_spec = BinaryFeatureSpec(
            ls_categories=ls_categories,
            positive_category=positive_category,
            feature_name=feature_name,
        )
        store = cls.from_category_indices_and_spec(
            feature_spec=feature_spec, id_to_category_idx=id_to_category_idx
        )
        return store

    @classmethod
    def from_category_indices_and_spec(
        cls: Type[BinaryCategoryStoreT],
        feature_spec: BinaryFeatureSpecProtocol,
        id_to_category_idx: Optional[Mapping[IdentifierType, int]] = None,
    ) -> BinaryCategoryStoreT:
        store = cls(feature_spec=feature_spec, id_to_category_idx=id_to_category_idx)
        return store

    @classmethod
    def from_categories(
        cls: Type[BinaryCategoryStoreT],
        ls_categories: List[str],
        positive_category: str,
        feature_name: Optional[str] = None,
        id_to_category: Optional[Mapping[IdentifierType, str]] = None,
    ) -> BinaryCategoryStoreT:
        feature_spec = BinaryFeatureSpec(
            ls_categories=ls_categories,
            positive_category=positive_category,
            feature_name=feature_name,
        )
        store = cls.from_categories_and_spec(
            feature_spec=feature_spec, id_to_category=id_to_category
        )
        return store

    @classmethod
    def from_categories_and_spec(
        cls: Type[BinaryCategoryStoreT],
        feature_spec: BinaryFeatureSpecProtocol,
        id_to_category: Optional[Mapping[IdentifierType, str]] = None,
    ) -> BinaryCategoryStoreT:
        if id_to_category is None:
            id_to_category = {}
        store = cls(feature_spec=feature_spec, id_to_category_idx=None)
        store.set_multiple_cat(id_to_category=id_to_category)
        return store

    @property
    def id_to_is_positive(self) -> Dict[IdentifierType, bool]:
        return {
            identifier: self.stored_category_is_positive(identifier=identifier)
            for identifier in self._data.keys()
        }

    @property
    def id_to_is_negative(self) -> Dict[IdentifierType, bool]:
        return {
            identifier: self.stored_category_is_negative(identifier=identifier)
            for identifier in self._data.keys()
        }

    def stored_category_is_positive(self, identifier: IdentifierType) -> bool:
        category = self.get_category(identifier=identifier)
        return self._feature_spec.is_positive(category=category)

    def stored_category_is_negative(self, identifier: IdentifierType) -> bool:
        category = self.get_category(identifier=identifier)
        return self._feature_spec.is_negative(category=category)
