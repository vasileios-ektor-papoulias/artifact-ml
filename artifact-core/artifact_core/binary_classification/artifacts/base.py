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
from artifact_core.libs.resources.classification.binary_classification_results import (
    BinaryClassificationResults,
)
from artifact_core.libs.resources.classification.binary_true_category_store import (
    BinaryTrueCategoryStore,
)
from artifact_core.libs.resources.classification.classification_results import (
    ClassificationResults,
)
from artifact_core.libs.resources.classification.true_category_store import TrueCategoryStore

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
    def _evaluate_binary_classification(
        self,
        true_category_store: BinaryTrueCategoryStore,
        classification_results: BinaryClassificationResults,
    ) -> ArtifactResultT: ...

    def _evaluate_classification(
        self,
        true_category_store: TrueCategoryStore[BinaryFeatureSpecProtocol],
        classification_results: ClassificationResults[BinaryFeatureSpecProtocol],
    ) -> ArtifactResultT:
        binary_true_category_store = BinaryTrueCategoryStore.build(
            ls_categories=true_category_store.ls_categories,
            id_to_category_idx=true_category_store.id_to_category_idx,
        )
        binary_classification_results = BinaryClassificationResults.build(
            ls_categories=classification_results.ls_categories,
            id_to_category=classification_results.prediction_store.id_to_category,
            id_to_logits=classification_results.distribution_store.id_to_logits,
        )
        return self._evaluate_binary_classification(
            true_category_store=binary_true_category_store,
            classification_results=binary_classification_results,
        )


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
