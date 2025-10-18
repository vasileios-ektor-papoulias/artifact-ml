from abc import abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

from artifact_core.base.artifact import Artifact
from artifact_core.base.artifact_dependencies import (
    ArtifactHyperparams,
    ArtifactResources,
    ArtifactResult,
)
from artifact_core.libs.resource_spec.categorical.protocol import CategoricalFeatureSpecProtocol
from artifact_core.libs.resource_validation.classification.resource_validator import (
    ClassificationResourceValidator,
)
from artifact_core.libs.resources.categorical.category_store.category_store import CategoryStore
from artifact_core.libs.resources.classification.classification_results import ClassificationResults

ArtifactHyperparamsT = TypeVar("ArtifactHyperparamsT", bound="ArtifactHyperparams")
ArtifactResultT = TypeVar("ArtifactResultT", bound=ArtifactResult)
CategoricalFeatureSpecProtocolT = TypeVar(
    "CategoricalFeatureSpecProtocolT", bound=CategoricalFeatureSpecProtocol
)
CategoryStoreT = TypeVar("CategoryStoreT", bound=CategoryStore)
ClassificationResultsT = TypeVar("ClassificationResultsT", bound=ClassificationResults)
ClassificationArtifactResourcesT = TypeVar(
    "ClassificationArtifactResourcesT", bound="ClassificationArtifactResources"
)


@dataclass(frozen=True)
class ClassificationArtifactResources(
    ArtifactResources, Generic[CategoryStoreT, ClassificationResultsT]
):
    true_category_store: CategoryStoreT
    classification_results: ClassificationResultsT


class ClassificationArtifact(
    Artifact[
        ClassificationArtifactResources[
            CategoryStoreT,
            ClassificationResultsT,
        ],
        ArtifactResultT,
        ArtifactHyperparamsT,
        CategoricalFeatureSpecProtocolT,
    ],
    Generic[
        ArtifactResultT,
        ArtifactHyperparamsT,
        CategoricalFeatureSpecProtocolT,
        CategoryStoreT,
        ClassificationResultsT,
    ],
):
    @abstractmethod
    def _evaluate_classification(
        self,
        true_category_store: CategoryStoreT,
        classification_results: ClassificationResultsT,
    ) -> ArtifactResultT: ...

    def _compute(
        self,
        resources: ClassificationArtifactResources[CategoryStoreT, ClassificationResultsT],
    ) -> ArtifactResultT:
        result = self._evaluate_classification(
            true_category_store=resources.true_category_store,
            classification_results=resources.classification_results,
        )
        return result

    def _validate(
        self, resources: ClassificationArtifactResources[CategoryStoreT, ClassificationResultsT]
    ) -> ClassificationArtifactResources[CategoryStoreT, ClassificationResultsT]:
        ClassificationResourceValidator.validate(
            true_category_store=resources.true_category_store,
            classification_results=resources.classification_results,
        )
        return resources
