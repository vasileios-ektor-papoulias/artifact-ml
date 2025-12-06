from typing import Generic, Mapping, Optional, TypeVar, Union

from artifact_core._base.primitives.artifact_type import ArtifactType
from artifact_core._base.typing.artifact_result import (
    Array,
    ArrayCollection,
    Plot,
    PlotCollection,
    Score,
    ScoreCollection,
)
from artifact_core._domains.classification.engine import ClassificationEngine
from artifact_core._libs.resource_specs.binary_classification.protocol import (
    BinaryClassSpecProtocol,
)
from artifact_core._libs.resources.binary_classification.class_store import BinaryClassStore
from artifact_core._libs.resources.binary_classification.classification_results import (
    BinaryClassificationResults,
)
from artifact_core._utils.collections.entity_store import IdentifierType
from artifact_core.binary_classification._resources import BinaryClassificationArtifactResources

ScoreTypeT = TypeVar("ScoreTypeT", bound=ArtifactType)
ArrayTypeT = TypeVar("ArrayTypeT", bound=ArtifactType)
PlotTypeT = TypeVar("PlotTypeT", bound=ArtifactType)
ScoreCollectionTypeT = TypeVar("ScoreCollectionTypeT", bound=ArtifactType)
ArrayCollectionTypeT = TypeVar("ArrayCollectionTypeT", bound=ArtifactType)
PlotCollectionTypeT = TypeVar("PlotCollectionTypeT", bound=ArtifactType)


class BinaryClassificationEngineBase(
    ClassificationEngine[
        BinaryClassStore,
        BinaryClassificationResults,
        BinaryClassSpecProtocol,
        ScoreTypeT,
        ArrayTypeT,
        PlotTypeT,
        ScoreCollectionTypeT,
        ArrayCollectionTypeT,
        PlotCollectionTypeT,
    ],
    Generic[
        ScoreTypeT,
        ArrayTypeT,
        PlotTypeT,
        ScoreCollectionTypeT,
        ArrayCollectionTypeT,
        PlotCollectionTypeT,
    ],
):
    def produce_binary_classification_score(
        self,
        score_type: Union[ScoreTypeT, str],
        true: Mapping[IdentifierType, str],
        predicted: Mapping[IdentifierType, str],
        probs_pos: Optional[Mapping[IdentifierType, float]] = None,
    ) -> Score:
        resources = BinaryClassificationArtifactResources.from_spec(
            class_spec=self._resource_spec,
            true=true,
            predicted=predicted,
            probs_pos=probs_pos,
        )
        return super().produce_score(score_type=score_type, resources=resources)

    def produce_binary_classification_array(
        self,
        array_type: Union[ArrayTypeT, str],
        true: Mapping[IdentifierType, str],
        predicted: Mapping[IdentifierType, str],
        probs_pos: Optional[Mapping[IdentifierType, float]] = None,
    ) -> Array:
        resources = BinaryClassificationArtifactResources.from_spec(
            class_spec=self._resource_spec,
            true=true,
            predicted=predicted,
            probs_pos=probs_pos,
        )
        return super().produce_array(array_type=array_type, resources=resources)

    def produce_binary_classification_plot(
        self,
        plot_type: Union[PlotTypeT, str],
        true: Mapping[IdentifierType, str],
        predicted: Mapping[IdentifierType, str],
        probs_pos: Optional[Mapping[IdentifierType, float]] = None,
    ) -> Plot:
        resources = BinaryClassificationArtifactResources.from_spec(
            class_spec=self._resource_spec,
            true=true,
            predicted=predicted,
            probs_pos=probs_pos,
        )
        return super().produce_plot(plot_type=plot_type, resources=resources)

    def produce_binary_classification_score_collection(
        self,
        score_collection_type: Union[ScoreCollectionTypeT, str],
        true: Mapping[IdentifierType, str],
        predicted: Mapping[IdentifierType, str],
        probs_pos: Optional[Mapping[IdentifierType, float]] = None,
    ) -> ScoreCollection:
        resources = BinaryClassificationArtifactResources.from_spec(
            class_spec=self._resource_spec,
            true=true,
            predicted=predicted,
            probs_pos=probs_pos,
        )
        return super().produce_score_collection(
            score_collection_type=score_collection_type, resources=resources
        )

    def produce_binary_classification_array_collection(
        self,
        array_collection_type: Union[ArrayCollectionTypeT, str],
        true: Mapping[IdentifierType, str],
        predicted: Mapping[IdentifierType, str],
        probs_pos: Optional[Mapping[IdentifierType, float]] = None,
    ) -> ArrayCollection:
        resources = BinaryClassificationArtifactResources.from_spec(
            class_spec=self._resource_spec,
            true=true,
            predicted=predicted,
            probs_pos=probs_pos,
        )
        return super().produce_array_collection(
            array_collection_type=array_collection_type, resources=resources
        )

    def produce_binary_classification_plot_collection(
        self,
        plot_collection_type: Union[PlotCollectionTypeT, str],
        true: Mapping[IdentifierType, str],
        predicted: Mapping[IdentifierType, str],
        probs_pos: Optional[Mapping[IdentifierType, float]] = None,
    ) -> PlotCollection:
        resources = BinaryClassificationArtifactResources.from_spec(
            class_spec=self._resource_spec,
            true=true,
            predicted=predicted,
            probs_pos=probs_pos,
        )
        return super().produce_plot_collection(
            plot_collection_type=plot_collection_type, resources=resources
        )
