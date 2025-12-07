from typing import Dict, Generic, Mapping, Optional, Sequence, Type, TypeVar, Union

import numpy as np

from artifact_core._base.typing.artifact_result import Array
from artifact_core._libs.resource_specs.classification.protocol import ClassSpecProtocol
from artifact_core._libs.tools.calculators.softmax_calculator import SoftmaxCalculator
from artifact_core._utils.collections.entity_store import EntityStore, IdentifierType

ClassDistribution = Union[Sequence[float], Array]


ClassSpecProtocolT = TypeVar("ClassSpecProtocolT", bound=ClassSpecProtocol)
ClassDistributionStoreT = TypeVar("ClassDistributionStoreT", bound="ClassDistributionStore")


class ClassDistributionStore(EntityStore[Array], Generic[ClassSpecProtocolT]):
    def __init__(
        self,
        class_spec: ClassSpecProtocolT,
        id_to_logits: Optional[Mapping[IdentifierType, Array]] = None,
    ):
        self._class_spec = class_spec
        super().__init__(initial=None)
        if id_to_logits is not None:
            self.set_logits_multiple(id_to_logits=id_to_logits)

    @classmethod
    def build_empty(
        cls: Type[ClassDistributionStoreT], class_spec: ClassSpecProtocolT
    ) -> ClassDistributionStoreT:
        return cls(class_spec=class_spec)

    def __repr__(self) -> str:
        return (
            f"ClassDistributionStore("
            f"label_name={self._class_spec.label_name!r}, "
            f"n_items={self.n_items}, n_classes={self._class_spec.n_classes})"
        )

    @property
    def class_spec(self) -> ClassSpecProtocolT:
        return self._class_spec

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
    def id_to_logits(self) -> Dict[IdentifierType, Array]:
        return {identifier: logits.copy() for identifier, logits in self._data.items()}

    @property
    def id_to_probs(self) -> Dict[IdentifierType, Array]:
        return {
            identifier: SoftmaxCalculator.compute_probs(logits=logits)
            for identifier, logits in self._data.items()
        }

    @property
    def arr_logits(self) -> Array:
        if not self._data:
            return np.empty((0, self._class_spec.n_classes), dtype=float)
        return np.vstack([logits for logits in self._data.values()])

    @property
    def arr_probs(self) -> Array:
        return SoftmaxCalculator.compute_probs(logits=self.arr_logits)

    def get_logits(self, identifier: IdentifierType) -> Array:
        self._require_id(identifier=identifier)
        arr_logits = self.get(identifier=identifier)
        return arr_logits.copy()

    def get_probs(self, identifier: IdentifierType) -> Array:
        logits = self.get_logits(identifier=identifier)
        return SoftmaxCalculator.compute_probs(logits)

    def get_logit(self, class_name: str, identifier: IdentifierType) -> float:
        logits = self.get_logits(identifier=identifier)
        class_name = self._validate_class(class_name=class_name)
        class_idx = self._class_spec.get_class_idx(class_name=class_name)
        logit = float(logits[class_idx])
        return logit

    def get_prob(self, class_name: str, identifier: IdentifierType) -> float:
        probs = self.get_probs(identifier=identifier)
        class_name = self._validate_class(class_name=class_name)
        class_idx = self._class_spec.get_class_idx(class_name=class_name)
        prob = float(probs[class_idx])
        return prob

    def set_logits(self, identifier: IdentifierType, logits: ClassDistribution) -> None:
        arr_logits = self._normalize_distn_shape(arr_distn=logits)
        self._validate_distn_array(arr_distn=arr_logits, n_classes=self._class_spec.n_classes)
        self.set(identifier=identifier, value=arr_logits)

    def set_probs(self, identifier: IdentifierType, probs: ClassDistribution) -> None:
        arr_probs = self._normalize_distn_shape(arr_distn=probs)
        self._validate_distn_array(arr_distn=arr_probs, n_classes=self._class_spec.n_classes)
        self._validate_probs(arr_probs=arr_probs)
        arr_logits = SoftmaxCalculator.compute_logits(probs=arr_probs)
        self.set_logits(identifier=identifier, logits=arr_logits)

    def set_nan(self, identifier: IdentifierType) -> None:
        arr_probs = np.full(self._class_spec.n_classes, np.nan, dtype=float)
        self.set_probs(identifier=identifier, probs=arr_probs)

    def set_concentrated_idx(self, identifier: IdentifierType, class_idx: int) -> None:
        self._require_class_idx(class_idx=class_idx)
        arr_probs = np.zeros(self._class_spec.n_classes, dtype=float)
        arr_probs[class_idx] = 1.0
        self.set_probs(identifier=identifier, probs=arr_probs)

    def set_concentrated(self, identifier: IdentifierType, class_name: str) -> None:
        class_name = self._validate_class(class_name=class_name)
        self._require_class(class_name=class_name)
        class_idx = self._class_spec.get_class_idx(class_name=class_name)
        self.set_concentrated_idx(identifier=identifier, class_idx=class_idx)

    def set_logits_multiple(self, id_to_logits: Mapping[IdentifierType, ClassDistribution]) -> None:
        for identifier, logits in id_to_logits.items():
            self.set_logits(identifier=identifier, logits=logits)

    def set_probs_multiple(self, id_to_probs: Mapping[IdentifierType, ClassDistribution]) -> None:
        for identifier, probs in id_to_probs.items():
            self.set_probs(identifier=identifier, probs=probs)

    def _require_id(self, identifier: IdentifierType) -> None:
        if identifier not in self._data:
            raise KeyError(
                f"Unknown identifier: {identifier!r}. Known identifiers: {list(self._data.keys())}"
            )

    def _require_class(self, class_name: str) -> None:
        if not self._class_spec.has_class(class_name):
            raise ValueError(
                f"Unknown class '{class_name}'. "
                f"Known classes (in order): {self._class_spec.class_names}"
            )

    def _require_class_idx(self, class_idx: int) -> None:
        if not isinstance(class_idx, int):
            raise TypeError("`class_idx` must be an integer.")
        if class_idx < 0 or class_idx >= self._class_spec.n_classes:
            raise IndexError(
                f"Class index out of range: "
                f"0 <= class_idx < {self._class_spec.n_classes}, "
                f"got {class_idx}."
            )

    @staticmethod
    def _validate_probs(arr_probs: Array, atol: float = 1e-8) -> None:
        if np.any(arr_probs < 0.0):
            raise ValueError("Probabilities must be non-negative.")
        total_mass = float(np.sum(arr_probs))
        if not np.isclose(total_mass, 1.0, atol=atol):
            raise ValueError(
                f"Probability distribution must sum to 1 (±{atol}). Got total mass: {total_mass}"
            )

    @staticmethod
    def _normalize_distn_shape(arr_distn: ClassDistribution) -> Array:
        arr_distn = np.asarray(arr_distn, dtype=float)
        if arr_distn.ndim == 1:
            return arr_distn.copy()
        if arr_distn.ndim == 2 and arr_distn.shape[0] == 1:
            return arr_distn.reshape(-1).copy()
        raise ValueError(
            f"Input must be a 1D array or a row vector of shape (1, n). "
            f"Got shape {arr_distn.shape}."
        )

    @staticmethod
    def _validate_distn_array(arr_distn: Array, n_classes: int) -> None:
        if np.any(np.isnan(arr_distn)):
            raise ValueError("Distribution array must not contain NaN.")
        if arr_distn.ndim != 1 or arr_distn.shape[0] != n_classes:
            raise ValueError(f"Vector must be shape ({n_classes},); got {arr_distn.shape}.")

    @staticmethod
    def _validate_class(class_name: str) -> str:
        return str(class_name)
