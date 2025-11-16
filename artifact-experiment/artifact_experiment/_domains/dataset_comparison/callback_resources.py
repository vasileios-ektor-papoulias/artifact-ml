from typing import TypeVar

from artifact_core.spi.resources import DatasetComparisonArtifactResources

from artifact_experiment._base.components.callbacks.artifact import ArtifactCallbackResources

DatasetTCov = TypeVar("DatasetTCov", covariant=True)

DatasetComparisonCallbackResources = ArtifactCallbackResources[
    DatasetComparisonArtifactResources[DatasetTCov]
]
