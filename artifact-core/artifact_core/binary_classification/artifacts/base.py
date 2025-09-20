from abc import abstractmethod
from typing import Dict, Generic, TypeVar

from matplotlib.figure import Figure
from numpy import ndarray

from artifact_core.base.artifact_dependencies import (
    ArtifactHyperparams,
    ArtifactResult,
)
from artifact_core.core.classification.artifact import (
    ClassificationArtifact,
    ClassificationArtifactResources,
)
from artifact_core.libs.resource_spec.binary.protocol import BinaryFeatureSpecProtocol
from artifact_core.libs.resources.categorical.category_store import BinaryCategoryStore
from artifact_core.libs.resources.classification.classification_results import (
    BinaryClassificationResults,
)

ArtifactResultT = TypeVar("ArtifactResultT", bound=ArtifactResult)
ArtifactHyperparamsT = TypeVar("ArtifactHyperparamsT", bound="ArtifactHyperparams")

BinaryClassificationArtifactResources = ClassificationArtifactResources[BinaryFeatureSpecProtocol]


class BinaryClassificationArtifact(
    ClassificationArtifact[
        ArtifactResultT,
        ArtifactHyperparamsT,
        BinaryFeatureSpecProtocol,
    ],
    Generic[ArtifactResultT, ArtifactHyperparamsT],
):
    @abstractmethod
    def _evaluate_classification(
        self,
        true_category_store: BinaryCategoryStore,
        classification_results: BinaryClassificationResults,
    ) -> ArtifactResultT: ...


BinaryClassificationScore = BinaryClassificationArtifact[float, ArtifactHyperparamsT]
BinaryClassificationArray = BinaryClassificationArtifact[ndarray, ArtifactHyperparamsT]
BinaryClassificationPlot = BinaryClassificationArtifact[Figure, ArtifactHyperparamsT]
BinaryClassificationScoreCollection = BinaryClassificationArtifact[
    Dict[str, float], ArtifactHyperparamsT
]
BinaryClassificationArrayCollection = BinaryClassificationArtifact[
    Dict[str, ndarray], ArtifactHyperparamsT
]
BinaryClassificationPlotCollection = BinaryClassificationArtifact[
    Dict[str, Figure], ArtifactHyperparamsT
]
