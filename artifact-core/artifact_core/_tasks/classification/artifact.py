from abc import abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

from artifact_core._base.contracts.hyperparams import ArtifactHyperparams
from artifact_core._base.contracts.resources import ArtifactResources
from artifact_core._base.core.artifact import Artifact
from artifact_core._base.types.artifact_result import ArtifactResult
from artifact_core._libs.resource_specs.classification.protocol import (
    CategoricalFeatureSpecProtocol,
)
from artifact_core._libs.resources.classification.category_store import CategoryStore
from artifact_core._libs.resources.classification.classification_results import (
    ClassificationResults,
)
from artifact_core._libs.validation.classification.resource_validator import (
    ClassificationResourceValidator,
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


CategoricalFeatureSpecProtocolT = TypeVar(
    "CategoricalFeatureSpecProtocolT", bound=CategoricalFeatureSpecProtocol
)
ArtifactHyperparamsT = TypeVar("ArtifactHyperparamsT", bound=ArtifactHyperparams)
ArtifactResultT = TypeVar("ArtifactResultT", bound=ArtifactResult)


class ClassificationArtifact(
    Artifact[
        ClassificationArtifactResources[
            CategoryStoreT,
            ClassificationResultsT,
        ],
        CategoricalFeatureSpecProtocolT,
        ArtifactHyperparamsT,
        ArtifactResultT,
    ],
    Generic[
        CategoryStoreT,
        ClassificationResultsT,
        CategoricalFeatureSpecProtocolT,
        ArtifactHyperparamsT,
        ArtifactResultT,
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
