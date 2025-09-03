from typing import Dict, Generic, TypeVar, Union

from matplotlib.figure import Figure
from numpy import ndarray

from artifact_core.base.artifact_dependencies import ResourceSpecProtocol
from artifact_core.base.engine import ArtifactEngine
from artifact_core.base.registry import ArtifactType
from artifact_core.core.classification.artifact import (
    ClassificationArtifactResources,
)

ScoreTypeT = TypeVar("ScoreTypeT", bound="ArtifactType")
ArrayTypeT = TypeVar("ArrayTypeT", bound="ArtifactType")
PlotTypeT = TypeVar("PlotTypeT", bound="ArtifactType")
ScoreCollectionTypeT = TypeVar("ScoreCollectionTypeT", bound="ArtifactType")
ArrayCollectionTypeT = TypeVar("ArrayCollectionTypeT", bound="ArtifactType")
PlotCollectionTypeT = TypeVar("PlotCollectionTypeT", bound="ArtifactType")
ResourceSpecProtocolT = TypeVar("ResourceSpecProtocolT", bound=ResourceSpecProtocol)
LabelsT = TypeVar("LabelsT")


class ClassifierEvaluationEngine(
    ArtifactEngine[
        ClassificationArtifactResources[LabelsT],
        ResourceSpecProtocolT,
        ScoreTypeT,
        ArrayTypeT,
        PlotTypeT,
        ScoreCollectionTypeT,
        ArrayCollectionTypeT,
        PlotCollectionTypeT,
    ],
    Generic[
        LabelsT,
        ResourceSpecProtocolT,
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
        labels_ground_truth: LabelsT,
        labels_predicted: LabelsT,
    ) -> float:
        resources = ClassificationArtifactResources[LabelsT](
            labels_ground_truth=labels_ground_truth, labels_predicted=labels_predicted
        )
        return super().produce_score(score_type=score_type, resources=resources)

    def produce_classifier_evaluation_array(
        self,
        array_type: Union[ArrayTypeT, str],
        labels_ground_truth: LabelsT,
        labels_predicted: LabelsT,
    ) -> ndarray:
        resources = ClassificationArtifactResources[LabelsT](
            labels_ground_truth=labels_ground_truth, labels_predicted=labels_predicted
        )
        return super().produce_array(array_type=array_type, resources=resources)

    def produce_classifier_evaluation_plot(
        self,
        plot_type: Union[PlotTypeT, str],
        labels_ground_truth: LabelsT,
        labels_predicted: LabelsT,
    ) -> Figure:
        resources = ClassificationArtifactResources[LabelsT](
            labels_ground_truth=labels_ground_truth, labels_predicted=labels_predicted
        )
        return super().produce_plot(plot_type=plot_type, resources=resources)

    def produce_classifier_evaluation_score_collection(
        self,
        score_collection_type: Union[ScoreCollectionTypeT, str],
        labels_ground_truth: LabelsT,
        labels_predicted: LabelsT,
    ) -> Dict[str, float]:
        resources = ClassificationArtifactResources[LabelsT](
            labels_ground_truth=labels_ground_truth, labels_predicted=labels_predicted
        )
        return super().produce_score_collection(
            score_collection_type=score_collection_type, resources=resources
        )

    def produce_classifier_evaluation_array_collection(
        self,
        array_collection_type: Union[ArrayCollectionTypeT, str],
        labels_ground_truth: LabelsT,
        labels_predicted: LabelsT,
    ) -> Dict[str, ndarray]:
        resources = ClassificationArtifactResources[LabelsT](
            labels_ground_truth=labels_ground_truth, labels_predicted=labels_predicted
        )
        return super().produce_array_collection(
            array_collection_type=array_collection_type, resources=resources
        )

    def produce_classifier_evaluation_plot_collection(
        self,
        plot_collection_type: Union[PlotCollectionTypeT, str],
        labels_ground_truth: LabelsT,
        labels_predicted: LabelsT,
    ) -> Dict[str, Figure]:
        resources = ClassificationArtifactResources[LabelsT](
            labels_ground_truth=labels_ground_truth, labels_predicted=labels_predicted
        )
        return super().produce_plot_collection(
            plot_collection_type=plot_collection_type, resources=resources
        )
