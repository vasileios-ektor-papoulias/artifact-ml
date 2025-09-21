from typing import Tuple

from artifact_core.libs.resource_spec.binary.protocol import BinaryFeatureSpecProtocol
from artifact_core.libs.resources.categorical.category_store import CategoryStore
from artifact_core.libs.resources.categorical.distribution_store import (
    CategoricalDistributionStore,
    IdentifierType,
)
from artifact_core.libs.resources.classification.classification_results import ClassificationResults


class BinaryClassificationResults(ClassificationResults[BinaryFeatureSpecProtocol]):
    def __init__(
        self,
        feature_spec: BinaryFeatureSpecProtocol,
        pred_store: CategoryStore[BinaryFeatureSpecProtocol],
        distn_store: CategoricalDistributionStore[BinaryFeatureSpecProtocol],
    ):
        super().__init__(feature_spec=feature_spec, pred_store=pred_store, distn_store=distn_store)
        self._pos_idx = self._idx_for_category(feature_spec.positive_category)
        self._neg_idx = self._idx_for_category(feature_spec.negative_category)

    @property
    def positive_category(self) -> str:
        return self._feature_spec.positive_category

    @property
    def negative_category(self) -> str:
        return self._feature_spec.negative_category

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
