from typing import (
    Dict,
    Generic,
    Iterable,
    List,
    Mapping,
    Optional,
    TypeVar,
)

import numpy as np

from artifact_core.libs.resource_spec.categorical.protocol import CategoricalFeatureSpecProtocol
from artifact_core.libs.resources.categorical.category_store.category_store import CategoryStore
from artifact_core.libs.resources.categorical.distribution_store.distribution_store import (
    CategoricalDistributionStore,
)
from artifact_core.libs.types.entity_store import IdentifierType

CategoricalFeatureSpecProtocolTCov = TypeVar(
    "CategoricalFeatureSpecProtocolTCov", bound=CategoricalFeatureSpecProtocol, covariant=True
)
CategoryStoreTCov = TypeVar("CategoryStoreTCov", bound=CategoryStore, covariant=True)
CategoricalDistributionStoreTCov = TypeVar(
    "CategoricalDistributionStoreTCov", bound=CategoricalDistributionStore, covariant=True
)
ClassificationResultsT = TypeVar("ClassificationResultsT", bound="ClassificationResults")


class ClassificationResults(
    Generic[CategoricalFeatureSpecProtocolTCov, CategoryStoreTCov, CategoricalDistributionStoreTCov]
):
    _feature_name = "predicted_category"

    def __init__(
        self,
        class_spec: CategoricalFeatureSpecProtocolTCov,
        pred_store: CategoryStoreTCov,
        distn_store: CategoricalDistributionStoreTCov,
    ):
        self._feature_spec = class_spec
        self._pred_store = pred_store
        self._distn_store = distn_store

    def __len__(self) -> int:
        return self.n_items

    def __repr__(self) -> str:
        return (
            f"ClassificationResults(feature_name={self.feature_name!r}, "
            f"n_items={self.n_items}, n_categories={self.n_categories})"
        )

    @property
    def feature_spec(self) -> CategoricalFeatureSpecProtocolTCov:
        return self._feature_spec

    @property
    def pred_store(self) -> CategoryStoreTCov:
        return self._pred_store

    @property
    def distn_store(self) -> CategoricalDistributionStoreTCov:
        return self._distn_store

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
    def n_items(self) -> int:
        return len(self._pred_store)

    @property
    def ids(self) -> Iterable[IdentifierType]:
        return self._pred_store.ids

    @property
    def id_to_logits(self) -> Dict[IdentifierType, np.ndarray]:
        return self._distn_store.id_to_logits

    @property
    def id_to_probs(self) -> Dict[IdentifierType, np.ndarray]:
        return self._distn_store.id_to_probs

    @property
    def id_to_predicted_category(self) -> Dict[IdentifierType, str]:
        return self._pred_store.id_to_category

    @property
    def id_to_predicted_category_idx(self) -> Dict[IdentifierType, int]:
        return self._pred_store.id_to_category_idx

    def get_predicted_index(self, identifier: IdentifierType) -> int:
        return self._pred_store.get_category_idx(identifier=identifier)

    def get_predicted_category(self, identifier: IdentifierType) -> str:
        return self._pred_store.get_category(identifier=identifier)

    def get_logits(self, identifier: IdentifierType) -> np.ndarray:
        return self._distn_store.get_logits(identifier=identifier)

    def get_probs(self, identifier: IdentifierType) -> np.ndarray:
        return self._distn_store.get_probs(identifier=identifier)

    def set_result(
        self,
        identifier: IdentifierType,
        predicted_category: str,
        logits: Optional[np.ndarray] = None,
    ) -> None:
        self._set_entry(identifier=identifier, category=predicted_category, logits=logits)

    def set_results(
        self,
        id_to_category: Mapping[IdentifierType, str],
        id_to_logits: Optional[Mapping[IdentifierType, np.ndarray]] = None,
    ) -> None:
        for identifier in id_to_category.keys():
            category = id_to_category[identifier]
            logits = id_to_logits.get(identifier) if id_to_logits is not None else None
            self._set_entry(identifier=identifier, category=category, logits=logits)

    def _set_entry(
        self,
        identifier: IdentifierType,
        category: str,
        logits: Optional[np.ndarray],
    ) -> None:
        self._pred_store.set_category(identifier=identifier, category=category)
        if logits is None:
            self._distn_store.set_concentrated(identifier=identifier, category=category)
        else:
            self._distn_store.set_logits(identifier=identifier, logits=logits)
