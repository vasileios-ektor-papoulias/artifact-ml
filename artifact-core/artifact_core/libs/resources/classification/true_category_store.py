from typing import List, Mapping, Optional, Type, TypeVar

from artifact_core.libs.resource_spec.binary.protocol import BinaryFeatureSpecProtocol
from artifact_core.libs.resource_spec.categorical.protocol import CategoricalFeatureSpecProtocol
from artifact_core.libs.resources.categorical.category_store import (
    CategoryStore,
    IdentifierType,
)

CategoricalFeatureSpecProtocolT = TypeVar(
    "CategoricalFeatureSpecProtocolT", bound=CategoricalFeatureSpecProtocol, covariant=True
)
TrueCategoryStoreT = TypeVar("TrueCategoryStoreT", bound="TrueCategoryStore")


class TrueCategoryStore(CategoryStore[CategoricalFeatureSpecProtocolT]):
    _feature_name = "true"

    @classmethod
    def build(
        cls: Type[TrueCategoryStoreT],
        ls_categories: List[str],
        id_to_category_idx: Optional[Mapping[IdentifierType, int]] = None,
        feature_name: Optional[str] = None,
    ) -> TrueCategoryStoreT:
        if feature_name is None:
            feature_name = cls._feature_name
        return super().build(
            feature_name=feature_name,
            ls_categories=ls_categories,
            id_to_category_idx=id_to_category_idx,
        )

    @classmethod
    def from_categories(
        cls: Type[TrueCategoryStoreT],
        ls_categories: List[str],
        id_to_category: Mapping[IdentifierType, str],
        feature_name: Optional[str] = None,
    ) -> TrueCategoryStoreT:
        if feature_name is None:
            feature_name = cls._feature_name
        return super().from_categories(
            feature_name=feature_name,
            ls_categories=ls_categories,
            id_to_category=id_to_category,
        )


BinaryTrueCategoryStore = TrueCategoryStore[BinaryFeatureSpecProtocol]
