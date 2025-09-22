from typing import List, Mapping, Optional, Type, TypeVar

import numpy as np

from artifact_core.libs.resource_spec.categorical.protocol import CategoricalFeatureSpecProtocol
from artifact_core.libs.resource_spec.categorical.spec import CategoricalFeatureSpec
from artifact_core.libs.resources.categorical.category_store.multiclass import (
    MulticlassCategoryStore,
)
from artifact_core.libs.resources.categorical.distribution_store.multiclass import (
    MulticlassDistributionStore,
)
from artifact_core.libs.resources.classification.classification_results import ClassificationResults
from artifact_core.libs.types.entity_store import IdentifierType

CategoricalFeatureSpecProtocolTCov = TypeVar(
    "CategoricalFeatureSpecProtocolTCov", bound=CategoricalFeatureSpecProtocol, covariant=True
)
MulticlassClassificationResultsT = TypeVar(
    "MulticlassClassificationResultsT", bound="MulticlassClassificationResults"
)


class MulticlassClassificationResults(
    ClassificationResults[
        CategoricalFeatureSpecProtocol, MulticlassCategoryStore, MulticlassDistributionStore
    ]
):
    @classmethod
    def build(
        cls: Type[MulticlassClassificationResultsT],
        ls_categories: List[str],
        id_to_category: Optional[Mapping[IdentifierType, str]] = None,
        id_to_logits: Optional[Mapping[IdentifierType, np.ndarray]] = None,
    ) -> MulticlassClassificationResultsT:
        feature_spec = CategoricalFeatureSpec(
            ls_categories=ls_categories, feature_name=cls._feature_name
        )
        store = cls.from_spec(
            feature_spec=feature_spec, id_to_category=id_to_category, id_to_logits=id_to_logits
        )
        return store

    @classmethod
    def from_spec(
        cls: Type[MulticlassClassificationResultsT],
        feature_spec: CategoricalFeatureSpecProtocol,
        id_to_category: Optional[Mapping[IdentifierType, str]] = None,
        id_to_logits: Optional[Mapping[IdentifierType, np.ndarray]] = None,
    ) -> MulticlassClassificationResultsT:
        pred_store = MulticlassCategoryStore(feature_spec=feature_spec)
        distn_store = MulticlassDistributionStore(feature_spec=feature_spec)
        classification_results = cls(
            class_spec=feature_spec,
            pred_store=pred_store,
            distn_store=distn_store,
        )
        if id_to_category is not None:
            classification_results.set_results(
                id_to_category=id_to_category, id_to_logits=id_to_logits or {}
            )
        return classification_results
