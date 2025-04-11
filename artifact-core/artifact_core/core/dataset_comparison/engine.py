from typing import Dict, Generic, TypeVar

from matplotlib.figure import Figure
from numpy import ndarray

from artifact_core.base.artifact_dependencies import DataSpecProtocol
from artifact_core.base.engine import ArtifactEngine
from artifact_core.base.registry import ArtifactType
from artifact_core.core.dataset_comparison.artifact import (
    DatasetComparisonArtifactResources,
)

scoreTypeT = TypeVar("scoreTypeT", bound="ArtifactType")
arrayTypeT = TypeVar("arrayTypeT", bound="ArtifactType")
plotTypeT = TypeVar("plotTypeT", bound="ArtifactType")
scoreCollectionTypeT = TypeVar("scoreCollectionTypeT", bound="ArtifactType")
arrayCollectionTypeT = TypeVar("arrayCollectionTypeT", bound="ArtifactType")
plotCollectionTypeT = TypeVar("plotCollectionTypeT", bound="ArtifactType")
dataSpecProtocolT = TypeVar("dataSpecProtocolT", bound=DataSpecProtocol)
datasetT = TypeVar("datasetT")


class DatasetComparisonEngine(
    ArtifactEngine[
        DatasetComparisonArtifactResources[datasetT],
        dataSpecProtocolT,
        scoreTypeT,
        arrayTypeT,
        plotTypeT,
        scoreCollectionTypeT,
        arrayCollectionTypeT,
        plotCollectionTypeT,
    ],
    Generic[
        datasetT,
        dataSpecProtocolT,
        scoreTypeT,
        arrayTypeT,
        plotTypeT,
        scoreCollectionTypeT,
        arrayCollectionTypeT,
        plotCollectionTypeT,
    ],
):
    def produce_dataset_comparison_score(
        self,
        score_type: scoreTypeT,
        dataset_real: datasetT,
        dataset_synthetic: datasetT,
    ) -> float:
        resources = DatasetComparisonArtifactResources[datasetT](
            dataset_real=dataset_real, dataset_synthetic=dataset_synthetic
        )
        return super().produce_score(score_type=score_type, resources=resources)

    def produce_dataset_comparison_array(
        self,
        array_type: arrayTypeT,
        dataset_real: datasetT,
        dataset_synthetic: datasetT,
    ) -> ndarray:
        resources = DatasetComparisonArtifactResources[datasetT](
            dataset_real=dataset_real, dataset_synthetic=dataset_synthetic
        )
        return super().produce_array(array_type=array_type, resources=resources)

    def produce_dataset_comparison_plot(
        self,
        plot_type: plotTypeT,
        dataset_real: datasetT,
        dataset_synthetic: datasetT,
    ) -> Figure:
        resources = DatasetComparisonArtifactResources[datasetT](
            dataset_real=dataset_real, dataset_synthetic=dataset_synthetic
        )
        return super().produce_plot(plot_type=plot_type, resources=resources)

    def produce_dataset_comparison_score_collection(
        self,
        score_collection_type: scoreCollectionTypeT,
        dataset_real: datasetT,
        dataset_synthetic: datasetT,
    ) -> Dict[str, float]:
        resources = DatasetComparisonArtifactResources[datasetT](
            dataset_real=dataset_real, dataset_synthetic=dataset_synthetic
        )
        return super().produce_score_collection(
            score_collection_type=score_collection_type, resources=resources
        )

    def produce_dataset_comparison_array_collection(
        self,
        array_collection_type: arrayCollectionTypeT,
        dataset_real: datasetT,
        dataset_synthetic: datasetT,
    ) -> Dict[str, ndarray]:
        resources = DatasetComparisonArtifactResources[datasetT](
            dataset_real=dataset_real, dataset_synthetic=dataset_synthetic
        )
        return super().produce_array_collection(
            array_collection_type=array_collection_type, resources=resources
        )

    def produce_dataset_comparison_plot_collection(
        self,
        plot_collection_type: plotCollectionTypeT,
        dataset_real: datasetT,
        dataset_synthetic: datasetT,
    ) -> Dict[str, Figure]:
        resources = DatasetComparisonArtifactResources[datasetT](
            dataset_real=dataset_real, dataset_synthetic=dataset_synthetic
        )
        return super().produce_plot_collection(
            plot_collection_type=plot_collection_type, resources=resources
        )
