from abc import abstractmethod
from dataclasses import dataclass
from typing import Generic, Tuple, TypeVar

from artifact_core.base.artifact import Artifact
from artifact_core.base.artifact_dependencies import (
    ArtifactHyperparams,
    ArtifactResources,
    ArtifactResult,
    DataSpecProtocol,
)

dataSpecProtocolT = TypeVar("dataSpecProtocolT", bound=DataSpecProtocol)
artifactHyperparamsT = TypeVar("artifactHyperparamsT", bound="ArtifactHyperparams")
datasetT = TypeVar("datasetT")
artifactResultT = TypeVar("artifactResultT", bound=ArtifactResult)


@dataclass(frozen=True)
class DatasetComparisonArtifactResources(ArtifactResources, Generic[datasetT]):
    dataset_real: datasetT
    dataset_synthetic: datasetT


class DatasetComparisonArtifact(
    Artifact[
        DatasetComparisonArtifactResources,
        artifactResultT,
        artifactHyperparamsT,
        dataSpecProtocolT,
    ],
    Generic[
        datasetT,
        artifactResultT,
        artifactHyperparamsT,
        dataSpecProtocolT,
    ],
):
    @abstractmethod
    def _compare_datasets(
        self, dataset_real: datasetT, dataset_synthetic: datasetT
    ) -> artifactResultT: ...

    @abstractmethod
    def _validate_datasets(
        self, dataset_real: datasetT, dataset_synthetic: datasetT
    ) -> Tuple[datasetT, datasetT]: ...

    def _compute(self, resources: DatasetComparisonArtifactResources[datasetT]) -> artifactResultT:
        result = self._compare_datasets(
            dataset_real=resources.dataset_real,
            dataset_synthetic=resources.dataset_synthetic,
        )
        return result

    def _validate(
        self, resources: DatasetComparisonArtifactResources[datasetT]
    ) -> DatasetComparisonArtifactResources[datasetT]:
        dataset_real_validated, dataset_synthetic_validated = self._validate_datasets(
            dataset_real=resources.dataset_real,
            dataset_synthetic=resources.dataset_synthetic,
        )
        resources_validated = DatasetComparisonArtifactResources[datasetT](
            dataset_real=dataset_real_validated,
            dataset_synthetic=dataset_synthetic_validated,
        )
        return resources_validated
