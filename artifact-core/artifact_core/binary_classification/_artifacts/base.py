from abc import abstractmethod
from typing import Dict, Generic, List, Mapping, Optional, Type, TypeVar

from matplotlib.figure import Figure

from artifact_core._base.contracts.hyperparams import ArtifactHyperparams
from artifact_core._base.types.artifact_result import Array, ArtifactResult
from artifact_core._libs.resource_specs.binary_classification.protocol import (
    BinaryFeatureSpecProtocol,
)
from artifact_core._libs.resource_specs.binary_classification.spec import BinaryFeatureSpec
from artifact_core._libs.resources.binary_classification.category_store import BinaryCategoryStore
from artifact_core._libs.resources.binary_classification.classification_results import (
    BinaryClassificationResults,
)
from artifact_core._libs.resources.tools.entity_store import IdentifierType
from artifact_core._tasks.classification.artifact import (
    ClassificationArtifact,
    ClassificationArtifactResources,
)

BinaryClassificationArtifactResourcesT = TypeVar(
    "BinaryClassificationArtifactResourcesT", bound="BinaryClassificationArtifactResources"
)


class BinaryClassificationArtifactResources(
    ClassificationArtifactResources[BinaryCategoryStore, BinaryClassificationResults]
):
    @classmethod
    def build(
        cls: Type[BinaryClassificationArtifactResourcesT],
        ls_categories: List[str],
        positive_category: str,
        true: Mapping[IdentifierType, str],
        predicted: Mapping[IdentifierType, str],
        probs_pos: Optional[Mapping[IdentifierType, float]] = None,
    ) -> BinaryClassificationArtifactResourcesT:
        class_spec = BinaryFeatureSpec(
            ls_categories=ls_categories, positive_category=positive_category
        )
        resources = cls.from_spec(
            class_spec=class_spec, true=true, predicted=predicted, probs_pos=probs_pos
        )
        return resources

    @classmethod
    def from_spec(
        cls: Type[BinaryClassificationArtifactResourcesT],
        class_spec: BinaryFeatureSpecProtocol,
        true: Mapping[IdentifierType, str],
        predicted: Mapping[IdentifierType, str],
        probs_pos: Optional[Mapping[IdentifierType, float]] = None,
    ) -> BinaryClassificationArtifactResourcesT:
        true_category_store = BinaryCategoryStore.from_categories_and_spec(
            feature_spec=class_spec, id_to_category=true
        )
        classification_results = BinaryClassificationResults.from_spec(
            class_spec=class_spec, id_to_category=predicted, id_to_prob_pos=probs_pos
        )
        resources = cls(
            true_category_store=true_category_store, classification_results=classification_results
        )
        return resources

    @classmethod
    def from_stores(
        cls: Type[BinaryClassificationArtifactResourcesT],
        true_category_store: BinaryCategoryStore,
        classification_results: BinaryClassificationResults,
    ) -> BinaryClassificationArtifactResourcesT:
        artifact_resources = cls(
            true_category_store=true_category_store, classification_results=classification_results
        )
        return artifact_resources


ArtifactResultT = TypeVar("ArtifactResultT", bound=ArtifactResult)
ArtifactHyperparamsT = TypeVar("ArtifactHyperparamsT", bound=ArtifactHyperparams)


class BinaryClassificationArtifact(
    ClassificationArtifact[
        BinaryCategoryStore,
        BinaryClassificationResults,
        BinaryFeatureSpecProtocol,
        ArtifactHyperparamsT,
        ArtifactResultT,
    ],
    Generic[ArtifactHyperparamsT, ArtifactResultT],
):
    @abstractmethod
    def _evaluate_classification(
        self,
        true_category_store: BinaryCategoryStore,
        classification_results: BinaryClassificationResults,
    ) -> ArtifactResultT: ...


BinaryClassificationScore = BinaryClassificationArtifact[ArtifactHyperparamsT, float]
BinaryClassificationArray = BinaryClassificationArtifact[ArtifactHyperparamsT, Array]
BinaryClassificationPlot = BinaryClassificationArtifact[ArtifactHyperparamsT, Figure]
BinaryClassificationScoreCollection = BinaryClassificationArtifact[
    ArtifactHyperparamsT, Dict[str, float]
]
BinaryClassificationArrayCollection = BinaryClassificationArtifact[
    ArtifactHyperparamsT, Dict[str, Array]
]
BinaryClassificationPlotCollection = BinaryClassificationArtifact[
    ArtifactHyperparamsT, Dict[str, Figure]
]
