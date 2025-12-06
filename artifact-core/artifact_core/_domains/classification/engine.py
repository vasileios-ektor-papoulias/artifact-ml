from typing import Generic, TypeVar, Union

from artifact_core._base.orchestration.engine import ArtifactEngine
from artifact_core._base.orchestration.registry import ArtifactType
from artifact_core._base.typing.artifact_result import (
    Array,
    ArrayCollection,
    Plot,
    PlotCollection,
    Score,
    ScoreCollection,
)
from artifact_core._domains.classification.artifact import ClassificationArtifactResources
from artifact_core._libs.resource_specs.classification.protocol import ClassSpecProtocol
from artifact_core._libs.resources.classification.class_store import ClassStore
from artifact_core._libs.resources.classification.classification_results import (
    ClassificationResults,
)

ClassSpecProtocolT = TypeVar("ClassSpecProtocolT", bound=ClassSpecProtocol)
ClassStoreT = TypeVar("ClassStoreT", bound=ClassStore)
ClassificationResultsT = TypeVar("ClassificationResultsT", bound=ClassificationResults)
ScoreTypeT = TypeVar("ScoreTypeT", bound=ArtifactType)
ArrayTypeT = TypeVar("ArrayTypeT", bound=ArtifactType)
PlotTypeT = TypeVar("PlotTypeT", bound=ArtifactType)
ScoreCollectionTypeT = TypeVar("ScoreCollectionTypeT", bound=ArtifactType)
ArrayCollectionTypeT = TypeVar("ArrayCollectionTypeT", bound=ArtifactType)
PlotCollectionTypeT = TypeVar("PlotCollectionTypeT", bound=ArtifactType)


class ClassificationEngine(
    ArtifactEngine[
        ClassificationArtifactResources[ClassStoreT, ClassificationResultsT],
        ClassSpecProtocolT,
        ScoreTypeT,
        ArrayTypeT,
        PlotTypeT,
        ScoreCollectionTypeT,
        ArrayCollectionTypeT,
        PlotCollectionTypeT,
    ],
    Generic[
        ClassStoreT,
        ClassificationResultsT,
        ClassSpecProtocolT,
        ScoreTypeT,
        ArrayTypeT,
        PlotTypeT,
        ScoreCollectionTypeT,
        ArrayCollectionTypeT,
        PlotCollectionTypeT,
    ],
):
    def produce_classification_score(
        self,
        score_type: Union[ScoreTypeT, str],
        true_class_store: ClassStoreT,
        classification_results: ClassificationResultsT,
    ) -> Score:
        resources = ClassificationArtifactResources(
            true_class_store=true_class_store, classification_results=classification_results
        )
        return super().produce_score(score_type=score_type, resources=resources)

    def produce_classification_array(
        self,
        array_type: Union[ArrayTypeT, str],
        true_class_store: ClassStoreT,
        classification_results: ClassificationResultsT,
    ) -> Array:
        resources = ClassificationArtifactResources(
            true_class_store=true_class_store, classification_results=classification_results
        )
        return super().produce_array(array_type=array_type, resources=resources)

    def produce_classification_plot(
        self,
        plot_type: Union[PlotTypeT, str],
        true_class_store: ClassStoreT,
        classification_results: ClassificationResultsT,
    ) -> Plot:
        resources = ClassificationArtifactResources(
            true_class_store=true_class_store, classification_results=classification_results
        )
        return super().produce_plot(plot_type=plot_type, resources=resources)

    def produce_classification_score_collection(
        self,
        score_collection_type: Union[ScoreCollectionTypeT, str],
        true_class_store: ClassStoreT,
        classification_results: ClassificationResultsT,
    ) -> ScoreCollection:
        resources = ClassificationArtifactResources(
            true_class_store=true_class_store, classification_results=classification_results
        )
        return super().produce_score_collection(
            score_collection_type=score_collection_type, resources=resources
        )

    def produce_classification_array_collection(
        self,
        array_collection_type: Union[ArrayCollectionTypeT, str],
        true_class_store: ClassStoreT,
        classification_results: ClassificationResultsT,
    ) -> ArrayCollection:
        resources = ClassificationArtifactResources(
            true_class_store=true_class_store, classification_results=classification_results
        )
        return super().produce_array_collection(
            array_collection_type=array_collection_type, resources=resources
        )

    def produce_classification_plot_collection(
        self,
        plot_collection_type: Union[PlotCollectionTypeT, str],
        true_class_store: ClassStoreT,
        classification_results: ClassificationResultsT,
    ) -> PlotCollection:
        resources = ClassificationArtifactResources(
            true_class_store=true_class_store, classification_results=classification_results
        )
        return super().produce_plot_collection(
            plot_collection_type=plot_collection_type, resources=resources
        )
