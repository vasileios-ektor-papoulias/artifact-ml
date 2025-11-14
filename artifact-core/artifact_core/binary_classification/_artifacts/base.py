from abc import abstractmethod
from typing import Generic, TypeVar

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
from artifact_core._domains.classification.artifact import ClassificationArtifact
from artifact_core._libs.resource_specs.binary_classification.protocol import (
    BinaryClassSpecProtocol,
)
from artifact_core._libs.resources.binary_classification.class_store import BinaryClassStore
from artifact_core._libs.resources.binary_classification.classification_results import (
    BinaryClassificationResults,
)

ArtifactResultT = TypeVar("ArtifactResultT", bound=ArtifactResult)
ArtifactHyperparamsT = TypeVar("ArtifactHyperparamsT", bound=ArtifactHyperparams)


class BinaryClassificationArtifact(
    ClassificationArtifact[
        BinaryClassStore,
        BinaryClassificationResults,
        BinaryClassSpecProtocol,
        ArtifactHyperparamsT,
        ArtifactResultT,
    ],
    Generic[ArtifactHyperparamsT, ArtifactResultT],
):
    @abstractmethod
    def _evaluate_classification(
        self,
        true_class_store: BinaryClassStore,
        classification_results: BinaryClassificationResults,
    ) -> ArtifactResultT: ...


BinaryClassificationScore = BinaryClassificationArtifact[ArtifactHyperparamsT, Score]
BinaryClassificationArray = BinaryClassificationArtifact[ArtifactHyperparamsT, Array]
BinaryClassificationPlot = BinaryClassificationArtifact[ArtifactHyperparamsT, Plot]
BinaryClassificationScoreCollection = BinaryClassificationArtifact[
    ArtifactHyperparamsT, ScoreCollection
]
BinaryClassificationArrayCollection = BinaryClassificationArtifact[
    ArtifactHyperparamsT, ArrayCollection
]
BinaryClassificationPlotCollection = BinaryClassificationArtifact[
    ArtifactHyperparamsT, PlotCollection
]
