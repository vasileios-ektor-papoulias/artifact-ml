from typing import Dict, Generic, TypeVar, Union

from matplotlib.figure import Figure

from artifact_core._base.contracts.resource_spec import ResourceSpecProtocol
from artifact_core._base.orchestration.engine import ArtifactEngine
from artifact_core._base.types.artifact_result import Array
from artifact_core._base.types.artifact_type import ArtifactType
from artifact_core._tasks.dataset_comparison.artifact import DatasetComparisonArtifactResources

ScoreTypeT = TypeVar("ScoreTypeT", bound=ArtifactType)
ArrayTypeT = TypeVar("ArrayTypeT", bound=ArtifactType)
PlotTypeT = TypeVar("PlotTypeT", bound=ArtifactType)
ScoreCollectionTypeT = TypeVar("ScoreCollectionTypeT", bound=ArtifactType)
ArrayCollectionTypeT = TypeVar("ArrayCollectionTypeT", bound=ArtifactType)
PlotCollectionTypeT = TypeVar("PlotCollectionTypeT", bound=ArtifactType)
ResourceSpecProtocolT = TypeVar("ResourceSpecProtocolT", bound=ResourceSpecProtocol)
DatasetT = TypeVar("DatasetT")


class DatasetComparisonEngine(
    ArtifactEngine[
        DatasetComparisonArtifactResources[DatasetT],
        ResourceSpecProtocolT,
        ScoreTypeT,
        ArrayTypeT,
        PlotTypeT,
        ScoreCollectionTypeT,
        ArrayCollectionTypeT,
        PlotCollectionTypeT,
    ],
    Generic[
        DatasetT,
        ResourceSpecProtocolT,
        ScoreTypeT,
        ArrayTypeT,
        PlotTypeT,
        ScoreCollectionTypeT,
        ArrayCollectionTypeT,
        PlotCollectionTypeT,
    ],
):
    def produce_dataset_comparison_score(
        self,
        score_type: Union[ScoreTypeT, str],
        dataset_real: DatasetT,
        dataset_synthetic: DatasetT,
    ) -> float:
        resources = DatasetComparisonArtifactResources[DatasetT](
            dataset_real=dataset_real, dataset_synthetic=dataset_synthetic
        )
        return super().produce_score(score_type=score_type, resources=resources)

    def produce_dataset_comparison_array(
        self,
        array_type: Union[ArrayTypeT, str],
        dataset_real: DatasetT,
        dataset_synthetic: DatasetT,
    ) -> Array:
        resources = DatasetComparisonArtifactResources[DatasetT](
            dataset_real=dataset_real, dataset_synthetic=dataset_synthetic
        )
        return super().produce_array(array_type=array_type, resources=resources)

    def produce_dataset_comparison_plot(
        self,
        plot_type: Union[PlotTypeT, str],
        dataset_real: DatasetT,
        dataset_synthetic: DatasetT,
    ) -> Figure:
        resources = DatasetComparisonArtifactResources[DatasetT](
            dataset_real=dataset_real, dataset_synthetic=dataset_synthetic
        )
        return super().produce_plot(plot_type=plot_type, resources=resources)

    def produce_dataset_comparison_score_collection(
        self,
        score_collection_type: Union[ScoreCollectionTypeT, str],
        dataset_real: DatasetT,
        dataset_synthetic: DatasetT,
    ) -> Dict[str, float]:
        resources = DatasetComparisonArtifactResources[DatasetT](
            dataset_real=dataset_real, dataset_synthetic=dataset_synthetic
        )
        return super().produce_score_collection(
            score_collection_type=score_collection_type, resources=resources
        )

    def produce_dataset_comparison_array_collection(
        self,
        array_collection_type: Union[ArrayCollectionTypeT, str],
        dataset_real: DatasetT,
        dataset_synthetic: DatasetT,
    ) -> Dict[str, Array]:
        resources = DatasetComparisonArtifactResources[DatasetT](
            dataset_real=dataset_real, dataset_synthetic=dataset_synthetic
        )
        return super().produce_array_collection(
            array_collection_type=array_collection_type, resources=resources
        )

    def produce_dataset_comparison_plot_collection(
        self,
        plot_collection_type: Union[PlotCollectionTypeT, str],
        dataset_real: DatasetT,
        dataset_synthetic: DatasetT,
    ) -> Dict[str, Figure]:
        resources = DatasetComparisonArtifactResources[DatasetT](
            dataset_real=dataset_real, dataset_synthetic=dataset_synthetic
        )
        return super().produce_plot_collection(
            plot_collection_type=plot_collection_type, resources=resources
        )
