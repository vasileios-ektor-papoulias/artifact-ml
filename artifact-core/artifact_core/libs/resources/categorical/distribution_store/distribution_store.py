from typing import Dict, Generic, List, Mapping, Optional, Sequence, TypeVar, Union

import numpy as np

from artifact_core.libs.resource_spec.categorical.protocol import CategoricalFeatureSpecProtocol
from artifact_core.libs.utils.calculators.softmax_calculator import SoftmaxCalculator
from artifact_core.libs.utils.data_structures.entity_store import EntityStore, IdentifierType

CategoricalDistribution = Union[Sequence[float], np.ndarray]
CategoricalFeatureSpecProtocolTCov = TypeVar(
    "CategoricalFeatureSpecProtocolTCov", bound=CategoricalFeatureSpecProtocol, covariant=True
)
CategoricalDistributionStoreT = TypeVar(
    "CategoricalDistributionStoreT", bound="CategoricalDistributionStore"
)


class CategoricalDistributionStore(
    EntityStore[np.ndarray], Generic[CategoricalFeatureSpecProtocolTCov]
):
    def __init__(
        self,
        feature_spec: CategoricalFeatureSpecProtocolTCov,
        id_to_logits: Optional[Mapping[IdentifierType, np.ndarray]] = None,
    ):
        self._feature_spec = feature_spec
        super().__init__(initial=None)
        if id_to_logits is not None:
            self.set_logits_multiple(id_to_logits=id_to_logits)

    def __repr__(self) -> str:
        return (
            f"CategoricalDistributionStore(feature_name={self._feature_spec.feature_name!r}, "
            f"n_items={self.n_items}, n_categories={self._feature_spec.n_categories})"
        )

    @property
    def feature_spec(self) -> CategoricalFeatureSpecProtocolTCov:
        return self._feature_spec

    @property
    def feature_name(self) -> str:
        return self._feature_spec.feature_name

    @property
    def ls_categories(self) -> List[str]:
        return self._feature_spec.ls_categories

    @property
    def n_categories(self) -> int:
        return self._feature_spec.n_categories

    @property
    def id_to_logits(self) -> Dict[IdentifierType, np.ndarray]:
        return {identifier: logits.copy() for identifier, logits in self._data.items()}

    @property
    def id_to_probs(self) -> Dict[IdentifierType, np.ndarray]:
        return {
            identifier: SoftmaxCalculator.compute_probs(logits=logits)
            for identifier, logits in self._data.items()
        }

    @property
    def arr_logits(self) -> np.ndarray:
        if not self._data:
            return np.empty((0, self._feature_spec.n_categories), dtype=float)
        return np.vstack([logits for logits in self._data.values()])

    @property
    def arr_probs(self) -> np.ndarray:
        return SoftmaxCalculator.compute_probs(logits=self.arr_logits)

    def get_logits(self, identifier: IdentifierType) -> np.ndarray:
        self._require_id(identifier=identifier)
        arr_logits = self.get(identifier=identifier)
        return arr_logits.copy()

    def get_probs(self, identifier: IdentifierType) -> np.ndarray:
        return SoftmaxCalculator.compute_probs(self.get_logits(identifier=identifier))

    def get_logit(self, category: str, identifier: IdentifierType) -> float:
        logits = self.get_logits(identifier=identifier)
        category = self._validate_category(category=category)
        category_idx = self._feature_spec.get_category_idx(category=category)
        logit = float(logits[category_idx])
        return logit

    def get_prob(self, category: str, identifier: IdentifierType) -> float:
        probs = self.get_probs(identifier=identifier)
        category = self._validate_category(category=category)
        category_idx = self._feature_spec.get_category_idx(category=category)
        prob = float(probs[category_idx])
        return prob

    def set_logits(self, identifier: IdentifierType, logits: CategoricalDistribution) -> None:
        arr_logits = self._normalize_distn_shape(arr_distn=logits)
        self._validate_distn_array(
            arr_distn=arr_logits, n_categories=self._feature_spec.n_categories
        )
        self.set(identifier=identifier, value=arr_logits)

    def set_probs(self, identifier: IdentifierType, probs: CategoricalDistribution) -> None:
        arr_probs = self._normalize_distn_shape(arr_distn=probs)
        self._validate_distn_array(
            arr_distn=arr_probs, n_categories=self._feature_spec.n_categories
        )
        self._validate_probs(arr_probs=arr_probs)
        arr_logits = SoftmaxCalculator.compute_logits(probs=arr_probs)
        self.set_logits(identifier=identifier, logits=arr_logits)

    def set_concentrated_idx(self, identifier: IdentifierType, category_idx: int) -> None:
        self._require_category_idx(category_idx=category_idx)
        arr_probs = np.zeros(self._feature_spec.n_categories, dtype=float)
        arr_probs[category_idx] = 1.0
        self.set_probs(identifier=identifier, probs=arr_probs)

    def set_concentrated(self, identifier: IdentifierType, category: str) -> None:
        category = self._validate_category(category=category)
        self._require_category(category=category)
        category_idx = self._feature_spec.get_category_idx(category=category)
        self.set_concentrated_idx(identifier=identifier, category_idx=category_idx)

    def set_logits_multiple(
        self, id_to_logits: Mapping[IdentifierType, CategoricalDistribution]
    ) -> None:
        for identifier, logits in id_to_logits.items():
            self.set_logits(identifier=identifier, logits=logits)

    def set_probs_multiple(
        self, id_to_probs: Mapping[IdentifierType, CategoricalDistribution]
    ) -> None:
        for identifier, probs in id_to_probs.items():
            self.set_probs(identifier=identifier, probs=probs)

    def _require_id(self, identifier: IdentifierType) -> None:
        if identifier not in self._data:
            raise KeyError(
                f"Unknown identifier: {identifier!r}. Known identifiers: {list(self._data.keys())}"
            )

    def _require_category(self, category: str) -> None:
        if category not in self._feature_spec.ls_categories:
            raise ValueError(
                f"Unknown category '{category}'. "
                f"Known categories (in order): {self._feature_spec.ls_categories}"
            )

    def _require_category_idx(self, category_idx: int) -> None:
        if not isinstance(category_idx, int):
            raise TypeError("`category_idx` must be an integer.")
        if category_idx < 0 or category_idx >= self._feature_spec.n_categories:
            raise IndexError(
                f"Category index out of range: "
                f"0 <= category_idx < {self._feature_spec.n_categories}, "
                f"got {category_idx}."
            )

    @staticmethod
    def _validate_probs(arr_probs: np.ndarray, atol: float = 1e-8) -> None:
        if np.any(arr_probs < 0.0):
            raise ValueError("Probabilities must be non-negative.")
        total_mass = float(np.sum(arr_probs))
        if not np.isclose(total_mass, 1.0, atol=atol):
            raise ValueError(
                f"Probability distribution must sum to 1 (±{atol}). Got total mass: {total_mass}"
            )

    @staticmethod
    def _normalize_distn_shape(arr_distn: CategoricalDistribution) -> np.ndarray:
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
    def _validate_distn_array(arr_distn: np.ndarray, n_categories: int) -> None:
        if np.any(np.isnan(arr_distn)):
            raise ValueError("Distribution array must not contain NaN.")
        if arr_distn.ndim != 1 or arr_distn.shape[0] != n_categories:
            raise ValueError(f"Vector must be shape ({n_categories},); got {arr_distn.shape}.")

    @staticmethod
    def _validate_category(category: str) -> str:
        return str(category)
