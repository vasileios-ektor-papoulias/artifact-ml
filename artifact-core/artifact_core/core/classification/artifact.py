from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict, Generic, List, Optional, Type, TypeVar

from numpy import ndarray

from artifact_core.base.artifact import Artifact
from artifact_core.base.artifact_dependencies import (
    ArtifactHyperparams,
    ArtifactResources,
    ArtifactResult,
)
from artifact_core.libs.resource_spec.categorical.protocol import CategoricalFeatureSpecProtocol
from artifact_core.libs.resource_validation.classification.classification_resource_validator import (
    ClassificationResourcesValidator,
)
from artifact_core.libs.resources.categorical.category_store import CategoryStore
from artifact_core.libs.resources.categorical.distribution_store import IdentifierType
from artifact_core.libs.resources.classification.classification_results import ClassificationResults

ArtifactHyperparamsT = TypeVar("ArtifactHyperparamsT", bound="ArtifactHyperparams")
ArtifactResultT = TypeVar("ArtifactResultT", bound=ArtifactResult)
CategoricalFeatureSpecProtocolT = TypeVar(
    "CategoricalFeatureSpecProtocolT", bound=CategoricalFeatureSpecProtocol
)
ClassificationArtifactResourcesT = TypeVar(
    "ClassificationArtifactResourcesT", bound="ClassificationArtifactResources"
)


@dataclass(frozen=True)
class ClassificationArtifactResources(ArtifactResources, Generic[CategoricalFeatureSpecProtocolT]):
    true_category_store: CategoryStore[CategoricalFeatureSpecProtocolT]
    classification_results: ClassificationResults[CategoricalFeatureSpecProtocolT]

    @classmethod
    def build(
        cls: Type[ClassificationArtifactResourcesT],
        ls_categories: List[str],
        true: Dict[IdentifierType, str],
        predicted: Dict[IdentifierType, str],
        logits: Optional[Dict[IdentifierType, ndarray]] = None,
    ) -> ClassificationArtifactResourcesT:
        true_category_store = CategoryStore[CategoricalFeatureSpecProtocolT].from_categories(
            feature_name="true", ls_categories=ls_categories, id_to_category=true
        )
        classification_results = ClassificationResults[CategoricalFeatureSpecProtocolT].build(
            ls_categories=ls_categories,
            id_to_category=predicted,
            id_to_logits=logits,
        )
        resources = cls(
            true_category_store=true_category_store, classification_results=classification_results
        )
        return resources


class ClassificationArtifact(
    Artifact[
        ClassificationArtifactResources[CategoricalFeatureSpecProtocolT],
        ArtifactResultT,
        ArtifactHyperparamsT,
        CategoricalFeatureSpecProtocolT,
    ],
    Generic[
        ArtifactResultT,
        ArtifactHyperparamsT,
        CategoricalFeatureSpecProtocolT,
    ],
):
    @abstractmethod
    def _evaluate_classification(
        self,
        true_category_store: CategoryStore[CategoricalFeatureSpecProtocolT],
        classification_results: ClassificationResults[CategoricalFeatureSpecProtocolT],
    ) -> ArtifactResultT: ...

    def _compute(
        self, resources: ClassificationArtifactResources[CategoricalFeatureSpecProtocolT]
    ) -> ArtifactResultT:
        result = self._evaluate_classification(
            true_category_store=resources.true_category_store,
            classification_results=resources.classification_results,
        )
        return result

    def _validate(
        self, resources: ClassificationArtifactResources[CategoricalFeatureSpecProtocolT]
    ) -> ClassificationArtifactResources[CategoricalFeatureSpecProtocolT]:
        true_category_store, classification_results = ClassificationResourcesValidator.validate(
            true_category_store=resources.true_category_store,
            classification_results=resources.classification_results,
        )
        return ClassificationArtifactResources[CategoricalFeatureSpecProtocolT](
            true_category_store=true_category_store,
            classification_results=classification_results,
        )
