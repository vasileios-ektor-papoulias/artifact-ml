from enum import Enum
from typing import Dict, Generic, Iterable, Mapping, Optional, Sequence, Type, TypeVar

from artifact_core._base.typing.artifact_result import Array
from artifact_core._libs.resource_specs.classification.protocol import ClassSpecProtocol
from artifact_core._libs.resources.classification.class_store import ClassStore
from artifact_core._libs.resources.classification.distribution_store import ClassDistributionStore
from artifact_core._utils.collections.entity_store import IdentifierType

ClassSpecProtocolT = TypeVar("ClassSpecProtocolT", bound=ClassSpecProtocol)
ClassStoreT = TypeVar("ClassStoreT", bound=ClassStore)
ClassDistributionStoreT = TypeVar("ClassDistributionStoreT", bound=ClassDistributionStore)
ClassificationResultsT = TypeVar("ClassificationResultsT", bound="ClassificationResults")


class DistributionInferenceType(Enum):
    NAN = "NAN"
    CONCENTRATED = "CONCENTRATED"


class ClassificationResults(Generic[ClassSpecProtocolT, ClassStoreT, ClassDistributionStoreT]):
    _distn_inference_type = DistributionInferenceType.CONCENTRATED

    def __init__(
        self,
        class_spec: ClassSpecProtocolT,
        pred_store: ClassStoreT,
        distn_store: ClassDistributionStoreT,
    ):
        self._class_spec = class_spec
        self._pred_store = pred_store
        self._distn_store = distn_store

    @classmethod
    def build_empty(
        cls: Type[ClassificationResultsT], class_spec: ClassSpecProtocolT
    ) -> ClassificationResultsT:
        pred_store = ClassStore.build_empty(class_spec=class_spec)
        distn_store = ClassDistributionStore.build_empty(class_spec=class_spec)
        return cls(class_spec=class_spec, pred_store=pred_store, distn_store=distn_store)

    def __len__(self) -> int:
        return self.n_items

    def __repr__(self) -> str:
        return (
            f"ClassificationResults(label_name={self.label_name!r}, "
            f"n_items={self.n_items}, n_classes={self.n_classes})"
        )

    @property
    def class_spec(self) -> ClassSpecProtocolT:
        return self._class_spec

    @property
    def pred_store(self) -> ClassStoreT:
        return self._pred_store

    @property
    def distn_store(self) -> ClassDistributionStoreT:
        return self._distn_store

    @property
    def label_name(self) -> str:
        return self._class_spec.label_name

    @property
    def class_names(self) -> Sequence[str]:
        return self._class_spec.class_names

    @property
    def n_classes(self) -> int:
        return self._class_spec.n_classes

    @property
    def n_items(self) -> int:
        return len(self._pred_store)

    @property
    def ids(self) -> Iterable[IdentifierType]:
        return self._pred_store.ids

    @property
    def id_to_logits(self) -> Dict[IdentifierType, Array]:
        return self._distn_store.id_to_logits

    @property
    def id_to_probs(self) -> Dict[IdentifierType, Array]:
        return self._distn_store.id_to_probs

    @property
    def id_to_predicted_class(self) -> Mapping[IdentifierType, str]:
        return self._pred_store.id_to_class_name

    @property
    def id_to_predicted_class_idx(self) -> Mapping[IdentifierType, int]:
        return self._pred_store.id_to_class_idx

    def get_predicted_index(self, identifier: IdentifierType) -> int:
        return self._pred_store.get_class_idx(identifier=identifier)

    def get_predicted_class(self, identifier: IdentifierType) -> str:
        return self._pred_store.get_class_name(identifier=identifier)

    def get_logits(self, identifier: IdentifierType) -> Array:
        return self._distn_store.get_logits(identifier=identifier)

    def get_probs(self, identifier: IdentifierType) -> Array:
        return self._distn_store.get_probs(identifier=identifier)

    def set_single(
        self,
        identifier: IdentifierType,
        predicted_class: str,
        logits: Optional[Array] = None,
    ) -> None:
        self._set_single(identifier=identifier, class_name=predicted_class, logits=logits)

    def set_multiple(
        self,
        id_to_class: Mapping[IdentifierType, str],
        id_to_logits: Optional[Mapping[IdentifierType, Array]] = None,
    ) -> None:
        for identifier in id_to_class.keys():
            class_name = id_to_class[identifier]
            if id_to_logits is not None:
                logits = id_to_logits.get(identifier, None)
            else:
                logits = None
            self._set_single(identifier=identifier, class_name=class_name, logits=logits)

    def _set_single(
        self,
        identifier: IdentifierType,
        class_name: str,
        logits: Optional[Array],
    ) -> None:
        self._set_prediction(identifier=identifier, class_name=class_name)
        self._set_distribution(identifier=identifier, class_name=class_name, logits=logits)

    def _set_prediction(self, identifier: IdentifierType, class_name: str) -> None:
        self._pred_store.set_class(identifier=identifier, class_name=class_name)

    def _set_distribution(
        self,
        identifier: IdentifierType,
        class_name: str,
        logits: Optional[Array],
    ) -> None:
        if logits is None:
            self._set_inferred_distribution(
                identifier=identifier,
                class_name=class_name,
                distribution_inference_type=self._distn_inference_type,
            )
        else:
            self._distn_store.set_logits(identifier=identifier, logits=logits)

    def _set_inferred_distribution(
        self,
        identifier: IdentifierType,
        class_name: str,
        distribution_inference_type: DistributionInferenceType,
    ) -> None:
        infer_type = self._distn_inference_type
        if infer_type is DistributionInferenceType.NAN:
            self._distn_store.set_nan(identifier=identifier)
        elif infer_type is DistributionInferenceType.CONCENTRATED:
            self._distn_store.set_concentrated(identifier=identifier, class_name=class_name)
        else:
            raise ValueError(
                f"Unrecognized distribution inference type: {distribution_inference_type}"
            )
