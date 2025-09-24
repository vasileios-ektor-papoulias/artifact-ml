from typing import Dict, List, Mapping, Optional, Tuple, Type, TypeVar

from artifact_core.libs.resource_spec.binary.protocol import BinaryFeatureSpecProtocol
from artifact_core.libs.resource_spec.binary.spec import BinaryFeatureSpec
from artifact_core.libs.resources.categorical.category_store.binary import BinaryCategoryStore
from artifact_core.libs.resources.categorical.distribution_store.binary import (
    BinaryDistributionStore,
)
from artifact_core.libs.resources.classification.classification_results import ClassificationResults
from artifact_core.libs.utils.calculators.sigmoid_calculator import SigmoidCalculator
from artifact_core.libs.utils.data_structures.entity_store import IdentifierType

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
        id_to_prob_pos: Optional[Mapping[IdentifierType, float]] = None,
    ) -> BinaryClassificationResultsT:
        class_spec = BinaryFeatureSpec(
            ls_categories=ls_categories,
            positive_category=positive_category,
            feature_name=cls._feature_name,
        )
        classification_results = cls.from_spec(
            class_spec=class_spec,
            id_to_category=id_to_category,
            id_to_prob_pos=id_to_prob_pos,
        )
        return classification_results

    @classmethod
    def from_spec(
        cls: Type[BinaryClassificationResultsT],
        class_spec: BinaryFeatureSpecProtocol,
        id_to_category: Mapping[IdentifierType, str],
        id_to_prob_pos: Optional[Mapping[IdentifierType, float]] = None,
    ) -> BinaryClassificationResultsT:
        pred_store = BinaryCategoryStore.from_categories_and_spec(feature_spec=class_spec)
        distn_store = BinaryDistributionStore.from_spec(feature_spec=class_spec)
        classification_results = cls(
            class_spec=class_spec,
            pred_store=pred_store,
            distn_store=distn_store,
        )
        if id_to_category is not None:
            classification_results.set_results_binary(
                id_to_category=id_to_category, id_to_prob_pos=id_to_prob_pos or {}
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
    def id_to_prob_pos(self) -> Dict[IdentifierType, float]:
        return self._distn_store.id_to_prob_pos

    @property
    def id_to_prob_neg(self) -> Dict[IdentifierType, float]:
        return self._distn_store.id_to_prob_neg

    @property
    def id_to_logit_pos(self) -> Dict[IdentifierType, float]:
        return self._distn_store.id_to_logit_pos

    @property
    def id_to_logit_neg(self) -> Dict[IdentifierType, float]:
        return self._distn_store.id_to_logit_neg

    def predicted_category_is_positive(self, identifier: IdentifierType) -> bool:
        return self._pred_store.stored_category_is_positive(identifier=identifier)

    def predicted_category_is_negative(self, identifier: IdentifierType) -> bool:
        return self._pred_store.stored_category_is_negative(identifier=identifier)

    def get_prob_pos(self, identifier: IdentifierType) -> float:
        return self._distn_store.get_prob_pos(identifier=identifier)

    def get_prob_neg(self, identifier: IdentifierType) -> float:
        return self._distn_store.get_prob_neg(identifier=identifier)

    def get_logit_pos(self, identifier: IdentifierType) -> float:
        return self._distn_store.get_logit_pos(identifier=identifier)

    def get_logit_neg(self, identifier: IdentifierType) -> float:
        return self._distn_store.get_logit_neg(identifier=identifier)

    def get_probs_pos_neg(self, identifier: IdentifierType) -> Tuple[float, float]:
        return self._distn_store.get_probs_pos_neg(identifier=identifier)

    def get_logits_pos_neg(self, identifier: IdentifierType) -> Tuple[float, float]:
        return self._distn_store.get_logits_pos_neg(identifier=identifier)

    def set_result_binary(
        self,
        identifier: IdentifierType,
        predicted_category: str,
        prob_pos: Optional[float] = None,
    ) -> None:
        self._set_entry_binary(
            identifier=identifier, category=predicted_category, prob_pos=prob_pos
        )

    def set_results_binary(
        self,
        id_to_category: Mapping[IdentifierType, str],
        id_to_prob_pos: Optional[Mapping[IdentifierType, float]] = None,
    ) -> None:
        for identifier in id_to_category.keys():
            category = id_to_category[identifier]
            prob_pos = id_to_prob_pos.get(identifier) if id_to_prob_pos is not None else None
            self._set_entry_binary(identifier=identifier, category=category, prob_pos=prob_pos)

    def _set_entry_binary(
        self,
        identifier: IdentifierType,
        category: str,
        prob_pos: Optional[float],
    ) -> None:
        self._pred_store.set_category(identifier=identifier, category=category)
        if prob_pos is not None:
            logit_pos = SigmoidCalculator.compute_logit(prob=prob_pos)
            self._distn_store.set_logit_pos(identifier=identifier, logit_pos=logit_pos)
        else:
            self._distn_store.set_concentrated(identifier=identifier, category=category)
