from typing import Dict, List, Mapping, Optional, Tuple, Type, TypeVar

import numpy as np

from artifact_core.libs.resource_spec.binary.protocol import BinaryFeatureSpecProtocol
from artifact_core.libs.resource_spec.binary.spec import BinaryFeatureSpec
from artifact_core.libs.resources.categorical.category_store.binary import BinaryCategoryStore
from artifact_core.libs.resources.categorical.distribution_store.binary import (
    BinaryDistributionStore,
)
from artifact_core.libs.resources.classification.classification_results import ClassificationResults
from artifact_core.libs.types.entity_store import IdentifierType

BinaryClassificationResultsT = TypeVar(
    "BinaryClassificationResultsT", bound="BinaryClassificationResults"
)


class BinaryClassificationResults(
    ClassificationResults[BinaryFeatureSpecProtocol, BinaryCategoryStore, BinaryDistributionStore]
):
    @classmethod
    def build(
        cls: Type[BinaryClassificationResultsT],
        ls_categories: List[str],
        positive_category: str,
        id_to_category: Mapping[IdentifierType, str],
        id_to_logits: Optional[Mapping[IdentifierType, np.ndarray]] = None,
    ) -> BinaryClassificationResultsT:
        class_spec = BinaryFeatureSpec(
            ls_categories=ls_categories,
            positive_category=positive_category,
            feature_name=cls._feature_name,
        )
        classification_results = cls.from_spec(
            class_spec=class_spec, id_to_category=id_to_category, id_to_logits=id_to_logits
        )
        return classification_results

    @classmethod
    def from_spec(
        cls: Type[BinaryClassificationResultsT],
        class_spec: BinaryFeatureSpecProtocol,
        id_to_category: Mapping[IdentifierType, str],
        id_to_logits: Optional[Mapping[IdentifierType, np.ndarray]] = None,
    ) -> BinaryClassificationResultsT:
        pred_store = BinaryCategoryStore.from_categories_and_spec(
            feature_spec=class_spec,
            id_to_category=id_to_category,
        )
        distn_store = BinaryDistributionStore.from_spec(
            feature_spec=class_spec, id_to_logits=id_to_logits
        )
        classification_results = cls(
            class_spec=class_spec,
            pred_store=pred_store,
            distn_store=distn_store,
        )
        if id_to_category is not None:
            classification_results.set_results_multiple(
                id_to_category=id_to_category, id_to_logits=id_to_logits or {}
            )
        return classification_results

    @property
    def positive_category(self) -> str:
        return self._feature_spec.positive_category

    @property
    def negative_category(self) -> str:
        return self._feature_spec.negative_category

    @property
    def id_to_predicted_positive(self) -> Dict[IdentifierType, bool]:
        return self._pred_store.id_to_is_positive

    @property
    def id_to_predicted_negative(self) -> Dict[IdentifierType, bool]:
        return self._pred_store.id_to_is_negative

    @property
    def id_to_positive_prob(self) -> Dict[IdentifierType, float]:
        return self._distn_store.id_to_positive_prob

    @property
    def id_to_negative_prob(self) -> Dict[IdentifierType, float]:
        return self._distn_store.id_to_negative_prob

    @property
    def id_to_positive_logit(self) -> Dict[IdentifierType, float]:
        return self._distn_store.id_to_positive_logit

    @property
    def id_to_negative_logit(self) -> Dict[IdentifierType, float]:
        return self._distn_store.id_to_negative_logit

    def predicted_category_is_positive(self, identifier: IdentifierType) -> bool:
        return self._pred_store.stored_category_is_positive(identifier=identifier)

    def predicted_category_is_negative(self, identifier: IdentifierType) -> bool:
        return self._pred_store.stored_category_is_negative(identifier=identifier)

    def get_positive_prob(self, identifier: IdentifierType) -> float:
        return self._distn_store.get_positive_prob(identifier=identifier)

    def get_negative_prob(self, identifier: IdentifierType) -> float:
        return self._distn_store.get_negative_prob(identifier=identifier)

    def get_positive_logit(self, identifier: IdentifierType) -> float:
        return self._distn_store.get_positive_logit(identifier=identifier)

    def get_negative_logit(self, identifier: IdentifierType) -> float:
        return self._distn_store.get_negative_logit(identifier=identifier)

    def get_pos_neg_probs(self, identifier: IdentifierType) -> Tuple[float, float]:
        return self._distn_store.get_pos_neg_probs(identifier=identifier)

    def get_pos_neg_logits(self, identifier: IdentifierType) -> Tuple[float, float]:
        return self._distn_store.get_pos_neg_logits(identifier=identifier)
