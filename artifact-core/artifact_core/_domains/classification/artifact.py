from abc import abstractmethod
from typing import Generic, TypeVar

from artifact_core._base.core.artifact import Artifact
from artifact_core._base.core.hyperparams import ArtifactHyperparams
from artifact_core._base.typing.artifact_result import (
    Array,
    ArrayCollection,
    ArtifactResult,
    Plot,
    PlotCollection,
    Score,
    ScoreCollection,
)
from artifact_core._domains.classification.resources import ClassificationArtifactResources
from artifact_core._libs.resource_specs.classification.protocol import ClassSpecProtocol
from artifact_core._libs.resources.classification.class_store import ClassStore
from artifact_core._libs.resources.classification.classification_results import (
    ClassificationResults,
)
from artifact_core._libs.validation.classification.resource_validator import (
    ClassificationResourceValidator,
)

CategoryStoreT = TypeVar("CategoryStoreT", bound=ClassStore)
ClassificationResultsT = TypeVar("ClassificationResultsT", bound=ClassificationResults)
ClassSpecProtocolT = TypeVar("ClassSpecProtocolT", bound=ClassSpecProtocol)
ArtifactHyperparamsT = TypeVar("ArtifactHyperparamsT", bound=ArtifactHyperparams)
ArtifactResultT = TypeVar("ArtifactResultT", bound=ArtifactResult)


class ClassificationArtifact(
    Artifact[
        ClassificationArtifactResources[
            CategoryStoreT,
            ClassificationResultsT,
        ],
        ClassSpecProtocolT,
        ArtifactHyperparamsT,
        ArtifactResultT,
    ],
    Generic[
        CategoryStoreT,
        ClassificationResultsT,
        ClassSpecProtocolT,
        ArtifactHyperparamsT,
        ArtifactResultT,
    ],
):
    @abstractmethod
    def _evaluate_classification(
        self,
        true_class_store: CategoryStoreT,
        classification_results: ClassificationResultsT,
    ) -> ArtifactResultT: ...

    def _compute(
        self,
        resources: ClassificationArtifactResources[CategoryStoreT, ClassificationResultsT],
    ) -> ArtifactResultT:
        result = self._evaluate_classification(
            true_class_store=resources.true_class_store,
            classification_results=resources.classification_results,
        )
        return result

    def _validate(
        self, resources: ClassificationArtifactResources[CategoryStoreT, ClassificationResultsT]
    ) -> ClassificationArtifactResources[CategoryStoreT, ClassificationResultsT]:
        ClassificationResourceValidator.validate(
            true_class_store=resources.true_class_store,
            classification_results=resources.classification_results,
        )
        return resources


ClassificationScore = ClassificationArtifact[
    CategoryStoreT, ClassificationResultsT, ClassSpecProtocolT, ArtifactHyperparamsT, Score
]
ClassificationArray = ClassificationArtifact[
    CategoryStoreT, ClassificationResultsT, ClassSpecProtocolT, ArtifactHyperparamsT, Array
]
ClassificationPlot = ClassificationArtifact[
    CategoryStoreT, ClassificationResultsT, ClassSpecProtocolT, ArtifactHyperparamsT, Plot
]
ClassificationScoreCollection = ClassificationArtifact[
    CategoryStoreT,
    ClassificationResultsT,
    ClassSpecProtocolT,
    ArtifactHyperparamsT,
    ScoreCollection,
]
ClassificationArrayCollection = ClassificationArtifact[
    CategoryStoreT,
    ClassificationResultsT,
    ClassSpecProtocolT,
    ArtifactHyperparamsT,
    ArrayCollection,
]
ClassificationPlotCollection = ClassificationArtifact[
    CategoryStoreT, ClassificationResultsT, ClassSpecProtocolT, ArtifactHyperparamsT, PlotCollection
]
