from typing import Dict, Mapping, Optional, Sequence, Tuple, Type, TypeVar

from artifact_core._libs.resource_specs.binary_classification.protocol import (
    BinaryClassSpecProtocol,
)
from artifact_core._libs.resource_specs.binary_classification.spec import BinaryClassSpec
from artifact_core._libs.resources.binary_classification.class_store import BinaryClassStore
from artifact_core._libs.resources.binary_classification.distribution_store import (
    BinaryDistributionStore,
)
from artifact_core._libs.resources.classification.classification_results import (
    ClassificationResults,
    DistributionInferenceType,
)
from artifact_core._libs.tools.calculators.sigmoid_calculator import SigmoidCalculator
from artifact_core._utils.collections.entity_store import IdentifierType

BinaryClassificationResultsT = TypeVar(
    "BinaryClassificationResultsT", bound="BinaryClassificationResults"
)


class BinaryClassificationResults(
    ClassificationResults[BinaryClassSpecProtocol, BinaryClassStore, BinaryDistributionStore]
):
    _distn_inference_type = DistributionInferenceType.CONCENTRATED
    _label_name = "label"

    @classmethod
    def build_empty(
        cls: Type[BinaryClassificationResultsT], class_spec: BinaryClassSpecProtocol
    ) -> BinaryClassificationResultsT:
        pred_store = BinaryClassStore.build_empty(class_spec=class_spec)
        distn_store = BinaryDistributionStore.build_empty(class_spec=class_spec)
        return cls(class_spec=class_spec, pred_store=pred_store, distn_store=distn_store)

    @classmethod
    def build(
        cls: Type[BinaryClassificationResultsT],
        class_names: Sequence[str],
        positive_class: str,
        id_to_class: Mapping[IdentifierType, str],
        id_to_prob_pos: Optional[Mapping[IdentifierType, float]] = None,
    ) -> BinaryClassificationResultsT:
        class_spec = BinaryClassSpec(
            class_names=class_names, positive_class=positive_class, label_name=cls._label_name
        )
        classification_results = cls.from_spec(
            class_spec=class_spec, id_to_class=id_to_class, id_to_prob_pos=id_to_prob_pos
        )
        return classification_results

    @classmethod
    def from_spec(
        cls: Type[BinaryClassificationResultsT],
        class_spec: BinaryClassSpecProtocol,
        id_to_class: Mapping[IdentifierType, str],
        id_to_prob_pos: Optional[Mapping[IdentifierType, float]] = None,
    ) -> BinaryClassificationResultsT:
        pred_store = BinaryClassStore.from_class_names_and_spec(class_spec=class_spec)
        distn_store = BinaryDistributionStore.from_spec(class_spec=class_spec)
        classification_results = cls(
            class_spec=class_spec, pred_store=pred_store, distn_store=distn_store
        )
        if id_to_class is not None:
            classification_results.set_multiple_binary(
                id_to_class=id_to_class, id_to_prob_pos=id_to_prob_pos or {}
            )
        return classification_results

    @property
    def positive_class(self) -> str:
        return self._class_spec.positive_class

    @property
    def negative_class(self) -> str:
        return self._class_spec.negative_class

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

    def predicted_class_is_positive(self, identifier: IdentifierType) -> bool:
        return self._pred_store.stored_class_is_positive(identifier=identifier)

    def predicted_class_is_negative(self, identifier: IdentifierType) -> bool:
        return self._pred_store.stored_class_is_negative(identifier=identifier)

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

    def set_single_binary(
        self,
        identifier: IdentifierType,
        predicted_class: str,
        prob_pos: Optional[float] = None,
    ) -> None:
        self._set_entry_binary(identifier=identifier, class_name=predicted_class, prob_pos=prob_pos)

    def set_multiple_binary(
        self,
        id_to_class: Mapping[IdentifierType, str],
        id_to_prob_pos: Optional[Mapping[IdentifierType, float]] = None,
    ) -> None:
        for identifier in id_to_class.keys():
            class_name = id_to_class[identifier]
            prob_pos = id_to_prob_pos.get(identifier, None) if id_to_prob_pos is not None else None
            self._set_entry_binary(identifier=identifier, class_name=class_name, prob_pos=prob_pos)

    def _set_entry_binary(
        self,
        identifier: IdentifierType,
        class_name: str,
        prob_pos: Optional[float] = None,
    ) -> None:
        self._set_prediction(identifier=identifier, class_name=class_name)
        self._set_distribution_binary(
            identifier=identifier, class_name=class_name, prob_pos=prob_pos
        )

    def _set_distribution_binary(
        self,
        identifier: IdentifierType,
        class_name: str,
        prob_pos: Optional[float] = None,
    ) -> None:
        if prob_pos is None:
            self._set_inferred_distribution(
                identifier=identifier,
                class_name=class_name,
                distribution_inference_type=self._distn_inference_type,
            )
        else:
            logit_pos = SigmoidCalculator.compute_logit(prob=prob_pos)
            self._distn_store.set_logit_pos(identifier=identifier, logit_pos=logit_pos)
