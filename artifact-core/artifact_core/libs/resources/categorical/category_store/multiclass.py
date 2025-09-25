from typing import List, Mapping, Optional, Type, TypeVar

from artifact_core.libs.resource_spec.categorical.protocol import CategoricalFeatureSpecProtocol
from artifact_core.libs.resource_spec.categorical.spec import CategoricalFeatureSpec
from artifact_core.libs.resources.categorical.category_store.category_store import CategoryStore
from artifact_core.libs.utils.data_structures.entity_store import IdentifierType

MulticlassCategoryStoreT = TypeVar("MulticlassCategoryStoreT", bound="MulticlassCategoryStore")


class MulticlassCategoryStore(CategoryStore[CategoricalFeatureSpecProtocol]):
    @classmethod
    def from_category_indices(
        cls: Type[MulticlassCategoryStoreT],
        ls_categories: List[str],
        feature_name: Optional[str] = None,
        id_to_category_idx: Optional[Mapping[IdentifierType, int]] = None,
    ) -> MulticlassCategoryStoreT:
        feature_spec = CategoricalFeatureSpec(
            ls_categories=ls_categories,
            feature_name=feature_name,
        )
        store = cls.from_category_indices_and_spec(
            feature_spec=feature_spec, id_to_category_idx=id_to_category_idx
        )
        return store

    @classmethod
    def from_category_indices_and_spec(
        cls: Type[MulticlassCategoryStoreT],
        feature_spec: CategoricalFeatureSpecProtocol,
        id_to_category_idx: Optional[Mapping[IdentifierType, int]] = None,
    ) -> MulticlassCategoryStoreT:
        store = cls(feature_spec=feature_spec, id_to_category_idx=id_to_category_idx)
        return store

    @classmethod
    def from_categories(
        cls: Type[MulticlassCategoryStoreT],
        ls_categories: List[str],
        feature_name: Optional[str] = None,
        id_to_category: Optional[Mapping[IdentifierType, str]] = None,
    ) -> MulticlassCategoryStoreT:
        feature_spec = CategoricalFeatureSpec(
            ls_categories=ls_categories,
            feature_name=feature_name,
        )
        store = cls.from_categories_and_spec(
            feature_spec=feature_spec, id_to_category=id_to_category
        )
        return store

    @classmethod
    def from_categories_and_spec(
        cls: Type[MulticlassCategoryStoreT],
        feature_spec: CategoricalFeatureSpecProtocol,
        id_to_category: Optional[Mapping[IdentifierType, str]] = None,
    ) -> MulticlassCategoryStoreT:
        if id_to_category is None:
            id_to_category = {}
        store = cls(feature_spec=feature_spec, id_to_category_idx=None)
        store.set_multiple_cat(id_to_category=id_to_category)
        return store
