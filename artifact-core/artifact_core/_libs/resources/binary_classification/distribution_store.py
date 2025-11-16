from typing import Dict, Mapping, Optional, Sequence, Tuple, Type, TypeVar

from artifact_core._base.typing.artifact_result import Array
from artifact_core._libs.resource_specs.binary_classification.protocol import (
    BinaryClassSpecProtocol,
)
from artifact_core._libs.resource_specs.binary_classification.spec import BinaryClassSpec
from artifact_core._libs.resources.classification.distribution_store import ClassDistributionStore
from artifact_core._libs.tools.calculators.binary_distribution_calculator import (
    BinaryDistributionCalculator,
)
from artifact_core._libs.tools.calculators.sigmoid_calculator import SigmoidCalculator
from artifact_core._utils.collections.entity_store import IdentifierType

BinaryClassDistributionStoreT = TypeVar(
    "BinaryClassDistributionStoreT", bound="BinaryDistributionStore"
)


class BinaryDistributionStore(ClassDistributionStore[BinaryClassSpecProtocol]):
    def __init__(
        self,
        class_spec: BinaryClassSpecProtocol,
        id_to_logits: Optional[Mapping[IdentifierType, Array]] = None,
    ):
        super().__init__(class_spec=class_spec, id_to_logits=id_to_logits)
        self._pos_idx = self._idx_for_class(class_name=class_spec.positive_class)
        self._neg_idx = self._idx_for_class(class_name=class_spec.negative_class)

    @classmethod
    def build(
        cls: Type[BinaryClassDistributionStoreT],
        class_names: Sequence[str],
        positive_class: str,
        label_name: Optional[str] = None,
        id_to_prob_pos: Optional[Mapping[IdentifierType, float]] = None,
    ) -> BinaryClassDistributionStoreT:
        class_spec = BinaryClassSpec(
            class_names=class_names,
            positive_class=positive_class,
            label_name=label_name,
        )
        store = cls.from_spec(class_spec=class_spec, id_to_prob_pos=id_to_prob_pos)
        return store

    @classmethod
    def from_spec(
        cls: Type[BinaryClassDistributionStoreT],
        class_spec: BinaryClassSpecProtocol,
        id_to_prob_pos: Optional[Mapping[IdentifierType, float]] = None,
    ) -> BinaryClassDistributionStoreT:
        store = cls(class_spec=class_spec)
        if id_to_prob_pos is not None:
            store.set_prob_pos_multiple(id_to_prob_pos=id_to_prob_pos)
        return store

    @property
    def positive_class(self) -> str:
        return self._class_spec.positive_class

    @property
    def negative_class(self) -> str:
        return self._class_spec.negative_class

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

    def _idx_for_class(self, class_name: str) -> int:
        try:
            return list(self._class_spec.class_names).index(class_name)
        except ValueError:
            raise ValueError(
                f"Class {class_name!r} not found. class_names={self._class_spec.class_names!r}"
            )
