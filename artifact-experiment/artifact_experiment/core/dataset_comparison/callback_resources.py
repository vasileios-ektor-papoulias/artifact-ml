from typing import TypeVar

from artifact_core._tasks.dataset_comparison.artifact import DatasetComparisonArtifactResources

from artifact_experiment.base.components.callbacks.artifact import ArtifactCallbackResources

DatasetTCov = TypeVar("DatasetTCov", covariant=True)

DatasetComparisonCallbackResources = ArtifactCallbackResources[
    DatasetComparisonArtifactResources[DatasetTCov]
]
