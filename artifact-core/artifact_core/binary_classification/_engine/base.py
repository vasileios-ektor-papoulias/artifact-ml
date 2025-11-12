from typing import Dict, Generic, Mapping, Optional, TypeVar, Union

from matplotlib.figure import Figure

from artifact_core._base.types.artifact_result import Array
from artifact_core._base.types.artifact_type import ArtifactType
from artifact_core._libs.resource_specs.binary_classification.protocol import (
    BinaryFeatureSpecProtocol,
)
from artifact_core._libs.resources.tools.entity_store import IdentifierType
from artifact_core._tasks.classification.engine import ClassificationEngine
from artifact_core.binary_classification._artifacts.base import (
    BinaryClassificationArtifactResources,
)

ScoreTypeT = TypeVar("ScoreTypeT", bound=ArtifactType)
ArrayTypeT = TypeVar("ArrayTypeT", bound=ArtifactType)
PlotTypeT = TypeVar("PlotTypeT", bound=ArtifactType)
ScoreCollectionTypeT = TypeVar("ScoreCollectionTypeT", bound=ArtifactType)
ArrayCollectionTypeT = TypeVar("ArrayCollectionTypeT", bound=ArtifactType)
PlotCollectionTypeT = TypeVar("PlotCollectionTypeT", bound=ArtifactType)


class BinaryClassificationEngineBase(
    ClassificationEngine[
        BinaryClassificationArtifactResources,
        BinaryFeatureSpecProtocol,
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
    def produce_classification_score(
        self,
        score_type: Union[ScoreTypeT, str],
        true: Mapping[IdentifierType, str],
        predicted: Mapping[IdentifierType, str],
        probs_pos: Optional[Mapping[IdentifierType, float]] = None,
    ) -> float:
        resources = BinaryClassificationArtifactResources.from_spec(
            class_spec=self._resource_spec,
            true=true,
            predicted=predicted,
            probs_pos=probs_pos,
        )
        return super().produce_score(score_type=score_type, resources=resources)

    def produce_classification_array(
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

    def produce_classification_plot(
        self,
        plot_type: Union[PlotTypeT, str],
        true: Mapping[IdentifierType, str],
        predicted: Mapping[IdentifierType, str],
        probs_pos: Optional[Mapping[IdentifierType, float]] = None,
    ) -> Figure:
        resources = BinaryClassificationArtifactResources.from_spec(
            class_spec=self._resource_spec,
            true=true,
            predicted=predicted,
            probs_pos=probs_pos,
        )
        return super().produce_plot(plot_type=plot_type, resources=resources)

    def produce_classification_score_collection(
        self,
        score_collection_type: Union[ScoreCollectionTypeT, str],
        true: Mapping[IdentifierType, str],
        predicted: Mapping[IdentifierType, str],
        probs_pos: Optional[Mapping[IdentifierType, float]] = None,
    ) -> Dict[str, float]:
        resources = BinaryClassificationArtifactResources.from_spec(
            class_spec=self._resource_spec,
            true=true,
            predicted=predicted,
            probs_pos=probs_pos,
        )
        return super().produce_score_collection(
            score_collection_type=score_collection_type, resources=resources
        )

    def produce_classification_array_collection(
        self,
        array_collection_type: Union[ArrayCollectionTypeT, str],
        true: Mapping[IdentifierType, str],
        predicted: Mapping[IdentifierType, str],
        probs_pos: Optional[Mapping[IdentifierType, float]] = None,
    ) -> Dict[str, Array]:
        resources = BinaryClassificationArtifactResources.from_spec(
            class_spec=self._resource_spec,
            true=true,
            predicted=predicted,
            probs_pos=probs_pos,
        )
        return super().produce_array_collection(
            array_collection_type=array_collection_type, resources=resources
        )

    def produce_classification_plot_collection(
        self,
        plot_collection_type: Union[PlotCollectionTypeT, str],
        true: Mapping[IdentifierType, str],
        predicted: Mapping[IdentifierType, str],
        probs_pos: Optional[Mapping[IdentifierType, float]] = None,
    ) -> Dict[str, Figure]:
        resources = BinaryClassificationArtifactResources.from_spec(
            class_spec=self._resource_spec,
            true=true,
            predicted=predicted,
            probs_pos=probs_pos,
        )
        return super().produce_plot_collection(
            plot_collection_type=plot_collection_type, resources=resources
        )
