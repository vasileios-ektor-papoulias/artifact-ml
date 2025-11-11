from typing import Dict, Generic, List, Mapping, Optional, TypeVar

from artifact_core._libs.resource_spec.categorical.protocol import CategoricalFeatureSpecProtocol
from artifact_core._libs.utils.data_structures.entity_store import EntityStore, IdentifierType

CategoricalFeatureSpecProtocolTCov = TypeVar(
    "CategoricalFeatureSpecProtocolTCov", bound=CategoricalFeatureSpecProtocol, covariant=True
)
CategoricalFeatureSpecProtocolT = TypeVar(
    "CategoricalFeatureSpecProtocolT", bound=CategoricalFeatureSpecProtocol
)
CategoryStoreT = TypeVar("CategoryStoreT", bound="CategoryStore")


class CategoryStore(EntityStore[int], Generic[CategoricalFeatureSpecProtocolTCov]):
    def __init__(
        self,
        feature_spec: CategoricalFeatureSpecProtocolTCov,
        id_to_category_idx: Optional[Mapping[IdentifierType, int]] = None,
    ):
        super().__init__(initial=None)
        self._feature_spec = feature_spec
        if id_to_category_idx is not None:
            self.set_multiple_idx(id_to_category_idx=id_to_category_idx)

    def __repr__(self) -> str:
        return (
            f"CategoryStore(feature_name={self._feature_spec.feature_name!r}, "
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
    def id_to_category_idx(self) -> Dict[IdentifierType, int]:
        return dict(self._data)

    @property
    def id_to_category(self) -> Dict[IdentifierType, str]:
        return {k: self._feature_spec.ls_categories[v] for k, v in self._data.items()}

    @property
    def ls_stored_indices(self) -> List[int]:
        return list(self._data.values())

    @property
    def ls_stored_categories(self) -> List[str]:
        cats = self._feature_spec.ls_categories
        return [cats[i] for i in self.ls_stored_indices]

    def get_category_idx(self, identifier: IdentifierType) -> int:
        self._require_id(identifier=identifier)
        category_idx = self.get(identifier=identifier)
        return category_idx

    def get_category(self, identifier: IdentifierType) -> str:
        category_idx = self.get_category_idx(identifier=identifier)
        category = self._feature_spec.ls_categories[category_idx]
        return category

    def set_category_idx(self, identifier: IdentifierType, category_idx: int) -> None:
        self._require_category_idx(category_idx=category_idx)
        self.set(identifier=identifier, value=category_idx)

    def set_category(self, identifier: IdentifierType, category: str) -> None:
        category = self._validate_category(category=category)
        self._require_category(category=category)
        category_idx = self._feature_spec.get_category_idx(category=category)
        self.set(identifier=identifier, value=category_idx)

    def set_multiple_idx(self, id_to_category_idx: Mapping[IdentifierType, int]) -> None:
        for identifier, category_idx in id_to_category_idx.items():
            self.set_category_idx(identifier=identifier, category_idx=category_idx)

    def set_multiple_cat(self, id_to_category: Mapping[IdentifierType, str]) -> None:
        for identifier, category in id_to_category.items():
            category = self._validate_category(category=category)
            self.set_category(identifier=identifier, category=category)

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
        if not (0 <= category_idx < self._feature_spec.n_categories):
            raise IndexError(
                f"Category index out of range: "
                f"0 <= category_idx < {self._feature_spec.n_categories}, "
                f"got {category_idx}."
            )

    @staticmethod
    def _validate_category(category: str) -> str:
        return str(category)
