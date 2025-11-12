from typing import List, Mapping, Optional, Type, TypeVar

from artifact_core._base.types.artifact_result import Array
from artifact_core._libs.resource_specs.classification.protocol import (
    CategoricalFeatureSpecProtocol,
)
from artifact_core._libs.resource_specs.classification.spec import CategoricalFeatureSpec
from artifact_core._libs.resources.classification.distribution_store import (
    CategoricalDistributionStore,
)
from artifact_core._libs.resources.tools.entity_store import IdentifierType

MulticlassDistributionStoreT = TypeVar(
    "MulticlassDistributionStoreT", bound="MulticlassDistributionStore"
)


class MulticlassDistributionStore(CategoricalDistributionStore[CategoricalFeatureSpecProtocol]):
    @classmethod
    def build(
        cls: Type[MulticlassDistributionStoreT],
        ls_categories: List[str],
        id_to_logits: Optional[Mapping[IdentifierType, Array]] = None,
        feature_name: Optional[str] = None,
    ) -> MulticlassDistributionStoreT:
        feature_spec = CategoricalFeatureSpec(
            ls_categories=ls_categories, feature_name=feature_name
        )
        store = cls.from_spec(feature_spec=feature_spec, id_to_logits=id_to_logits)
        return store

    @classmethod
    def from_spec(
        cls: Type[MulticlassDistributionStoreT],
        feature_spec: CategoricalFeatureSpecProtocol,
        id_to_logits: Optional[Mapping[IdentifierType, Array]] = None,
    ) -> MulticlassDistributionStoreT:
        store = cls(feature_spec=feature_spec, id_to_logits=id_to_logits)
        return store
