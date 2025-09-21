from abc import abstractmethod
from typing import Dict, Generic, List, Optional, Type, TypeVar

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
from artifact_core.libs.resource_spec.binary.spec import BinaryFeatureSpec
from artifact_core.libs.resources.categorical.category_store.binary import BinaryCategoryStore
from artifact_core.libs.resources.classification.binary_classification_results import (
    BinaryClassificationResults,
)
from artifact_core.libs.types.entity_store import IdentifierType

ArtifactResultT = TypeVar("ArtifactResultT", bound=ArtifactResult)
ArtifactHyperparamsT = TypeVar("ArtifactHyperparamsT", bound=ArtifactHyperparams)
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
        true: Dict[IdentifierType, str],
        predicted: Dict[IdentifierType, str],
        logits: Optional[Dict[IdentifierType, ndarray]] = None,
    ) -> BinaryClassificationArtifactResourcesT:
        class_spec = BinaryFeatureSpec(
            ls_categories=ls_categories, positive_category=positive_category
        )
        resources = cls.from_spec(
            class_spec=class_spec, true=true, predicted=predicted, ilogits=logits
        )
        return resources

    @classmethod
    def from_spec(
        cls: Type[BinaryClassificationArtifactResourcesT],
        class_spec: BinaryFeatureSpecProtocol,
        true: Dict[IdentifierType, str],
        predicted: Dict[IdentifierType, str],
        logits: Optional[Dict[IdentifierType, ndarray]] = None,
    ) -> BinaryClassificationArtifactResourcesT:
        true_category_store = BinaryCategoryStore.from_categories_and_spec(
            feature_spec=class_spec, id_to_category=true
        )
        classification_results = BinaryClassificationResults.from_spec(
            class_spec=class_spec,
            id_to_category=predicted,
            id_to_logits=logits,
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


class BinaryClassificationArtifact(
    ClassificationArtifact[
        ArtifactResultT,
        ArtifactHyperparamsT,
        BinaryFeatureSpecProtocol,
        BinaryCategoryStore,
        BinaryClassificationResults,
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
