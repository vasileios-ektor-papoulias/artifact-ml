from dataclasses import dataclass
from typing import Generic, Type, TypeVar

from artifact_core.core.dataset_comparison.artifact import DatasetComparisonArtifactResources

from artifact_experiment.base.callbacks.artifact import ArtifactCallbackResources

DatasetT = TypeVar("DatasetT")
DatasetComparisonCallbackResourcesT = TypeVar(
    "DatasetComparisonCallbackResourcesT", bound="DatasetComparisonCallbackResources"
)


@dataclass
class DatasetComparisonCallbackResources(
    ArtifactCallbackResources[DatasetComparisonArtifactResources[DatasetT]], Generic[DatasetT]
):
    @classmethod
    def build(
        cls: Type[DatasetComparisonCallbackResourcesT],
        dataset_real: DatasetT,
        dataset_synthetic: DatasetT,
    ) -> DatasetComparisonCallbackResourcesT:
        artifact_resources = DatasetComparisonArtifactResources(
            dataset_real=dataset_real, dataset_synthetic=dataset_synthetic
        )
        callback_resources = cls(artifact_resources=artifact_resources)
        return callback_resources
