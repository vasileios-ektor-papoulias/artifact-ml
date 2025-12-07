from abc import abstractmethod
from typing import Generic, TypeVar

from artifact_core._base.core.artifact import Artifact
from artifact_core._base.core.hyperparams import ArtifactHyperparams
from artifact_core._base.typing.artifact_result import ArtifactResult
from artifact_core._domains.classification.resources import ClassificationArtifactResources
from artifact_core._libs.resource_specs.classification.protocol import ClassSpecProtocol
from artifact_core._libs.resources.classification.class_store import ClassStore
from artifact_core._libs.resources.classification.classification_results import (
    ClassificationResults,
)
from artifact_core._libs.validation.classification.resource_validator import (
    ClassificationResourceValidator,
)

ClassStoreT = TypeVar("ClassStoreT", bound=ClassStore)
ClassificationResultsT = TypeVar("ClassificationResultsT", bound=ClassificationResults)
ClassSpecProtocolT = TypeVar("ClassSpecProtocolT", bound=ClassSpecProtocol)
ArtifactHyperparamsT = TypeVar("ArtifactHyperparamsT", bound=ArtifactHyperparams)
ArtifactResultT = TypeVar("ArtifactResultT", bound=ArtifactResult)


class ClassificationArtifact(
    Artifact[
        ClassificationArtifactResources[ClassStoreT, ClassificationResultsT],
        ClassSpecProtocolT,
        ArtifactHyperparamsT,
        ArtifactResultT,
    ],
    Generic[
        ClassStoreT,
        ClassificationResultsT,
        ClassSpecProtocolT,
        ArtifactHyperparamsT,
        ArtifactResultT,
    ],
):
    @abstractmethod
    def _evaluate_classification(
        self,
        true_class_store: ClassStoreT,
        classification_results: ClassificationResultsT,
    ) -> ArtifactResultT: ...

    def _compute(
        self,
        resources: ClassificationArtifactResources[ClassStoreT, ClassificationResultsT],
    ) -> ArtifactResultT:
        result = self._evaluate_classification(
            true_class_store=resources.true_class_store,
            classification_results=resources.classification_results,
        )
        return result

    def _validate(
        self, resources: ClassificationArtifactResources[ClassStoreT, ClassificationResultsT]
    ) -> ClassificationArtifactResources[ClassStoreT, ClassificationResultsT]:
        ClassificationResourceValidator.validate(
            true_class_store=resources.true_class_store,
            classification_results=resources.classification_results,
        )
        return resources
