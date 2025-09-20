from typing import (
    Generic,
    Iterable,
    List,
    Mapping,
    Optional,
    Type,
    TypeVar,
)

import numpy as np

from artifact_core.libs.resource_spec.binary.protocol import BinaryFeatureSpecProtocol
from artifact_core.libs.resource_spec.categorical.protocol import CategoricalFeatureSpecProtocol
from artifact_core.libs.resource_spec.categorical.spec import CategoricalFeatureSpec
from artifact_core.libs.resources.categorical.category_store import CategoryStore
from artifact_core.libs.resources.categorical.distribution_store import (
    CategoricalDistributionStore,
    IdentifierType,
)

CategoricalFeatureSpecProtocolT = TypeVar(
    "CategoricalFeatureSpecProtocolT", bound=CategoricalFeatureSpecProtocol, covariant=True
)
ClassificationResultsT = TypeVar("ClassificationResultsT", bound="ClassificationResults")


class ClassificationResults(Generic[CategoricalFeatureSpecProtocolT]):
    _categorical_feature_name = "classes"

    def __init__(
        self,
        feature_spec: CategoricalFeatureSpecProtocol,
        id_to_category: Optional[Mapping[IdentifierType, str]] = None,
        id_to_logits: Optional[Mapping[IdentifierType, np.ndarray]] = None,
    ):
        self._feature_spec = feature_spec
        self._pred_store = CategoryStore[CategoricalFeatureSpecProtocolT](feature_spec=feature_spec)
        self._distn_store = CategoricalDistributionStore[CategoricalFeatureSpecProtocolT](
            feature_spec=feature_spec
        )
        if id_to_category is not None:
            self.set_results_multiple(
                id_to_category=id_to_category, id_to_logits=id_to_logits or {}
            )

    @classmethod
    def build(
        cls: Type[ClassificationResultsT],
        ls_categories: List[str],
        id_to_category: Optional[Mapping[IdentifierType, str]] = None,
        id_to_logits: Optional[Mapping[IdentifierType, np.ndarray]] = None,
    ) -> ClassificationResultsT:
        feature_spec = CategoricalFeatureSpec(
            feature_name=cls._categorical_feature_name, ls_categories=ls_categories
        )
        return cls(
            feature_spec=feature_spec,
            id_to_category=id_to_category,
            id_to_logits=id_to_logits,
        )

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
    def prediction_store(self) -> CategoryStore:
        return self._pred_store

    @property
    def distribution_store(self) -> CategoricalDistributionStore:
        return self._distn_store

    @property
    def ids(self) -> Iterable[IdentifierType]:
        return self._pred_store.ids

    @property
    def n_items(self) -> int:
        return len(self._pred_store)

    def __len__(self) -> int:
        return self.n_items

    def __repr__(self) -> str:
        return (
            f"ClassificationResults(feature_name={self.feature_name!r}, "
            f"n_items={self.n_items}, n_categories={self.n_categories})"
        )

    def get_predicted_index(self, identifier: IdentifierType) -> int:
        return self._pred_store.get_category_idx(identifier)

    def get_predicted_category(self, identifier: IdentifierType) -> str:
        return self._pred_store.get_category(identifier)

    def get_logits(self, identifier: IdentifierType) -> np.ndarray:
        return self._distn_store.get_logits(identifier)

    def get_probs(self, identifier: IdentifierType) -> np.ndarray:
        return self._distn_store.get_probs(identifier)

    def set_result(
        self,
        identifier: IdentifierType,
        predicted_category: str,
        logits: Optional[np.ndarray] = None,
    ) -> None:
        self._set_entry(identifier=identifier, category=predicted_category, logits=logits)

    def set_results_multiple(
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


BinaryClassificationResults = ClassificationResults[BinaryFeatureSpecProtocol]
