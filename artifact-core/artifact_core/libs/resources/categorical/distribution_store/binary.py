from typing import Dict, List, Mapping, Optional, Tuple, Type, TypeVar

import numpy as np

from artifact_core.libs.resource_spec.binary.protocol import BinaryFeatureSpecProtocol
from artifact_core.libs.resource_spec.binary.spec import BinaryFeatureSpec
from artifact_core.libs.resources.categorical.distribution_store.distribution_store import (
    CategoricalDistributionStore,
)
from artifact_core.libs.types.entity_store import IdentifierType

BinaryDistributionStoreT = TypeVar("BinaryDistributionStoreT", bound="BinaryDistributionStore")


class BinaryDistributionStore(CategoricalDistributionStore[BinaryFeatureSpecProtocol]):
    def __init__(
        self,
        feature_spec: BinaryFeatureSpecProtocol,
        id_to_logits: Optional[Mapping[IdentifierType, np.ndarray]] = None,
    ):
        super().__init__(feature_spec=feature_spec, id_to_logits=id_to_logits)
        self._pos_idx = self._idx_for_category(category=feature_spec.positive_category)
        self._neg_idx = self._idx_for_category(category=feature_spec.negative_category)

    @classmethod
    def build(
        cls: Type[BinaryDistributionStoreT],
        ls_categories: List[str],
        positive_category: str,
        feature_name: Optional[str] = None,
        id_to_logits: Optional[Mapping[IdentifierType, np.ndarray]] = None,
    ) -> BinaryDistributionStoreT:
        feature_spec = BinaryFeatureSpec(
            ls_categories=ls_categories,
            positive_category=positive_category,
            feature_name=feature_name,
        )
        store = cls.from_spec(feature_spec=feature_spec, id_to_logits=id_to_logits)
        return store

    @classmethod
    def from_spec(
        cls: Type[BinaryDistributionStoreT],
        feature_spec: BinaryFeatureSpecProtocol,
        id_to_logits: Optional[Mapping[IdentifierType, np.ndarray]] = None,
    ) -> BinaryDistributionStoreT:
        store = cls(feature_spec=feature_spec, id_to_logits=id_to_logits)
        return store

    @property
    def positive_category(self) -> str:
        return self._feature_spec.positive_category

    @property
    def negative_category(self) -> str:
        return self._feature_spec.negative_category

    @property
    def id_to_positive_prob(self) -> Dict[IdentifierType, float]:
        return {
            identifier: self.get_positive_prob(identifier=identifier) for identifier in self.ids
        }

    @property
    def id_to_negative_prob(self) -> Dict[IdentifierType, float]:
        return {
            identifier: self.get_negative_prob(identifier=identifier) for identifier in self.ids
        }

    @property
    def id_to_positive_logit(self) -> Dict[IdentifierType, float]:
        return {
            identifier: self.get_positive_logit(identifier=identifier) for identifier in self.ids
        }

    @property
    def id_to_negative_logit(self) -> Dict[IdentifierType, float]:
        return {
            identifier: self.get_negative_logit(identifier=identifier) for identifier in self.ids
        }

    def get_positive_prob(self, identifier: IdentifierType) -> float:
        probs = self.get_probs(identifier=identifier)
        return float(probs[self._pos_idx])

    def get_negative_prob(self, identifier: IdentifierType) -> float:
        probs = self.get_probs(identifier=identifier)
        return float(probs[self._neg_idx])

    def get_positive_logit(self, identifier: IdentifierType) -> float:
        logits = self.get_logits(identifier=identifier)
        return float(logits[self._pos_idx])

    def get_negative_logit(self, identifier: IdentifierType) -> float:
        logits = self.get_logits(identifier=identifier)
        return float(logits[self._neg_idx])

    def get_pos_neg_probs(self, identifier: IdentifierType) -> Tuple[float, float]:
        probs = self.get_probs(identifier=identifier)
        return float(probs[self._pos_idx]), float(probs[self._neg_idx])

    def get_pos_neg_logits(self, identifier: IdentifierType) -> Tuple[float, float]:
        logits = self.get_logits(identifier=identifier)
        return float(logits[self._pos_idx]), float(logits[self._neg_idx])

    def _idx_for_category(self, category: str) -> int:
        try:
            return self._feature_spec.ls_categories.index(category)
        except ValueError:
            raise ValueError(
                f"Category {category!r} not found. "
                f"ls_categories={self._feature_spec.ls_categories!r}"
            )
