from typing import Dict, Generic, Optional, TypeVar, Union

from matplotlib.figure import Figure
from numpy import ndarray

from artifact_core.base.engine import ArtifactEngine
from artifact_core.base.registry import ArtifactType
from artifact_core.core.classification.artifact import ClassificationArtifactResources
from artifact_core.libs.resource_spec.categorical.protocol import CategoricalFeatureSpecProtocol
from artifact_core.libs.types.entity_store import IdentifierType

ScoreTypeT = TypeVar("ScoreTypeT", bound="ArtifactType")
ArrayTypeT = TypeVar("ArrayTypeT", bound="ArtifactType")
PlotTypeT = TypeVar("PlotTypeT", bound="ArtifactType")
ScoreCollectionTypeT = TypeVar("ScoreCollectionTypeT", bound="ArtifactType")
ArrayCollectionTypeT = TypeVar("ArrayCollectionTypeT", bound="ArtifactType")
PlotCollectionTypeT = TypeVar("PlotCollectionTypeT", bound="ArtifactType")
CategoricalFeatureSpecProtocolT = TypeVar(
    "CategoricalFeatureSpecProtocolT", bound=CategoricalFeatureSpecProtocol
)


class ClassifierEvaluationEngine(
    ArtifactEngine[
        ClassificationArtifactResources[CategoricalFeatureSpecProtocolT],
        CategoricalFeatureSpecProtocolT,
        ScoreTypeT,
        ArrayTypeT,
        PlotTypeT,
        ScoreCollectionTypeT,
        ArrayCollectionTypeT,
        PlotCollectionTypeT,
    ],
    Generic[
        CategoricalFeatureSpecProtocolT,
        ScoreTypeT,
        ArrayTypeT,
        PlotTypeT,
        ScoreCollectionTypeT,
        ArrayCollectionTypeT,
        PlotCollectionTypeT,
    ],
):
    def produce_classifier_evaluation_score(
        self,
        score_type: Union[ScoreTypeT, str],
        true: Dict[IdentifierType, str],
        predicted: Dict[IdentifierType, str],
        logits: Optional[Dict[IdentifierType, ndarray]] = None,
    ) -> float:
        resources = ClassificationArtifactResources[CategoricalFeatureSpecProtocolT].build(
            ls_categories=self._resource_spec.ls_categories,
            true=true,
            predicted=predicted,
            logits=logits,
        )
        return super().produce_score(score_type=score_type, resources=resources)

    def produce_classifier_evaluation_array(
        self,
        array_type: Union[ArrayTypeT, str],
        true: Dict[IdentifierType, str],
        predicted: Dict[IdentifierType, str],
        logits: Optional[Dict[IdentifierType, ndarray]] = None,
    ) -> ndarray:
        resources = ClassificationArtifactResources[CategoricalFeatureSpecProtocolT].build(
            ls_categories=self._resource_spec.ls_categories,
            true=true,
            predicted=predicted,
            logits=logits,
        )
        return super().produce_array(array_type=array_type, resources=resources)

    def produce_classifier_evaluation_plot(
        self,
        plot_type: Union[PlotTypeT, str],
        true: Dict[IdentifierType, str],
        predicted: Dict[IdentifierType, str],
        logits: Optional[Dict[IdentifierType, ndarray]] = None,
    ) -> Figure:
        resources = ClassificationArtifactResources[CategoricalFeatureSpecProtocolT].build(
            ls_categories=self._resource_spec.ls_categories,
            true=true,
            predicted=predicted,
            logits=logits,
        )
        return super().produce_plot(plot_type=plot_type, resources=resources)

    def produce_classifier_evaluation_score_collection(
        self,
        score_collection_type: Union[ScoreCollectionTypeT, str],
        true: Dict[IdentifierType, str],
        predicted: Dict[IdentifierType, str],
        logits: Optional[Dict[IdentifierType, ndarray]] = None,
    ) -> Dict[str, float]:
        resources = ClassificationArtifactResources[CategoricalFeatureSpecProtocolT].build(
            ls_categories=self._resource_spec.ls_categories,
            true=true,
            predicted=predicted,
            logits=logits,
        )
        return super().produce_score_collection(
            score_collection_type=score_collection_type, resources=resources
        )

    def produce_classifier_evaluation_array_collection(
        self,
        array_collection_type: Union[ArrayCollectionTypeT, str],
        true: Dict[IdentifierType, str],
        predicted: Dict[IdentifierType, str],
        logits: Optional[Dict[IdentifierType, ndarray]] = None,
    ) -> Dict[str, ndarray]:
        resources = ClassificationArtifactResources[CategoricalFeatureSpecProtocolT].build(
            ls_categories=self._resource_spec.ls_categories,
            true=true,
            predicted=predicted,
            logits=logits,
        )
        return super().produce_array_collection(
            array_collection_type=array_collection_type, resources=resources
        )

    def produce_classifier_evaluation_plot_collection(
        self,
        plot_collection_type: Union[PlotCollectionTypeT, str],
        true: Dict[IdentifierType, str],
        predicted: Dict[IdentifierType, str],
        logits: Optional[Dict[IdentifierType, ndarray]] = None,
    ) -> Dict[str, Figure]:
        resources = ClassificationArtifactResources[CategoricalFeatureSpecProtocolT].build(
            ls_categories=self._resource_spec.ls_categories,
            true=true,
            predicted=predicted,
            logits=logits,
        )
        return super().produce_plot_collection(
            plot_collection_type=plot_collection_type, resources=resources
        )
