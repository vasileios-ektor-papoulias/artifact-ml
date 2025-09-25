from typing import Dict, List, Mapping, Optional, Tuple, Type, TypeVar

import numpy as np

from artifact_core.libs.resource_spec.binary.protocol import BinaryFeatureSpecProtocol
from artifact_core.libs.resource_spec.binary.spec import BinaryFeatureSpec
from artifact_core.libs.resources.categorical.distribution_store.distribution_store import (
    CategoricalDistributionStore,
)
from artifact_core.libs.utils.calculators.binary_distribution_calculator import (
    BinaryDistributionCalculator,
)
from artifact_core.libs.utils.calculators.sigmoid_calculator import SigmoidCalculator
from artifact_core.libs.utils.data_structures.entity_store import IdentifierType

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
        id_to_prob_pos: Optional[Mapping[IdentifierType, float]] = None,
    ) -> BinaryDistributionStoreT:
        feature_spec = BinaryFeatureSpec(
            ls_categories=ls_categories,
            positive_category=positive_category,
            feature_name=feature_name,
        )
        store = cls.from_spec(feature_spec=feature_spec, id_to_prob_pos=id_to_prob_pos)
        return store

    @classmethod
    def from_spec(
        cls: Type[BinaryDistributionStoreT],
        feature_spec: BinaryFeatureSpecProtocol,
        id_to_prob_pos: Optional[Mapping[IdentifierType, float]] = None,
    ) -> BinaryDistributionStoreT:
        store = cls(feature_spec=feature_spec)
        if id_to_prob_pos is not None:
            store.set_prob_pos_multiple(id_to_prob_pos=id_to_prob_pos)
        return store

    @property
    def positive_category(self) -> str:
        return self._feature_spec.positive_category

    @property
    def negative_category(self) -> str:
        return self._feature_spec.negative_category

    @property
    def id_to_prob_pos(self) -> Dict[IdentifierType, float]:
        return {identifier: self.get_prob_pos(identifier=identifier) for identifier in self.ids}

    @property
    def id_to_prob_neg(self) -> Dict[IdentifierType, float]:
        return {identifier: self.get_prob_neg(identifier=identifier) for identifier in self.ids}

    @property
    def id_to_logit_pos(self) -> Dict[IdentifierType, float]:
        return {identifier: self.get_logit_pos(identifier=identifier) for identifier in self.ids}

    @property
    def id_to_logit_neg(self) -> Dict[IdentifierType, float]:
        return {identifier: self.get_logit_neg(identifier=identifier) for identifier in self.ids}

    def get_prob_pos(self, identifier: IdentifierType) -> float:
        probs = self.get_probs(identifier=identifier)
        return float(probs[self._pos_idx])

    def get_prob_neg(self, identifier: IdentifierType) -> float:
        probs = self.get_probs(identifier=identifier)
        return float(probs[self._neg_idx])

    def get_logit_pos(self, identifier: IdentifierType) -> float:
        logits = self.get_logits(identifier=identifier)
        return float(logits[self._pos_idx])

    def get_logit_neg(self, identifier: IdentifierType) -> float:
        logits = self.get_logits(identifier=identifier)
        return float(logits[self._neg_idx])

    def get_probs_pos_neg(self, identifier: IdentifierType) -> Tuple[float, float]:
        probs = self.get_probs(identifier=identifier)
        return float(probs[self._pos_idx]), float(probs[self._neg_idx])

    def get_logits_pos_neg(self, identifier: IdentifierType) -> Tuple[float, float]:
        logits = self.get_logits(identifier=identifier)
        return float(logits[self._pos_idx]), float(logits[self._neg_idx])

    def set_logit_pos(self, identifier: IdentifierType, logit_pos: float) -> None:
        prob_pos = SigmoidCalculator.compute_prob(logit=logit_pos)
        self.set_prob_pos(identifier=identifier, prob_pos=prob_pos)

    def set_prob_pos(self, identifier: IdentifierType, prob_pos: float) -> None:
        arr_probs = BinaryDistributionCalculator.compute_probs(
            prob_pos=prob_pos, pos_idx=self._pos_idx
        )
        self.set_probs(identifier=identifier, probs=arr_probs)

    def set_prob_pos_multiple(self, id_to_prob_pos: Mapping[IdentifierType, float]) -> None:
        for identifier, prob_pos in id_to_prob_pos.items():
            self.set_prob_pos(identifier=identifier, prob_pos=prob_pos)

    def set_logit_pos_multiple(self, id_to_logit_pos: Mapping[IdentifierType, float]) -> None:
        for identifier, logit_pos in id_to_logit_pos.items():
            self.set_logit_pos(identifier=identifier, logit_pos=logit_pos)

    def _idx_for_category(self, category: str) -> int:
        try:
            return self._feature_spec.ls_categories.index(category)
        except ValueError:
            raise ValueError(
                f"Category {category!r} not found. "
                f"ls_categories={self._feature_spec.ls_categories!r}"
            )
