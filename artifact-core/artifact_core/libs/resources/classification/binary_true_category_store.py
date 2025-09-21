from typing import Dict, TypeVar

from artifact_core.libs.resource_spec.binary.protocol import BinaryFeatureSpecProtocol
from artifact_core.libs.resources.categorical.category_store import (
    IdentifierType,
)
from artifact_core.libs.resources.classification.true_category_store import TrueCategoryStore

BinaryTrueCategoryStoreT = TypeVar("BinaryTrueCategoryStoreT", bound="BinaryTrueCategoryStore")


class BinaryTrueCategoryStore(TrueCategoryStore[BinaryFeatureSpecProtocol]):
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
