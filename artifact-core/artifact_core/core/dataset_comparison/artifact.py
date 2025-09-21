from abc import abstractmethod
from dataclasses import dataclass
from typing import Generic, Tuple, TypeVar

from artifact_core.base.artifact import Artifact
from artifact_core.base.artifact_dependencies import (
    ArtifactHyperparams,
    ArtifactResources,
    ArtifactResult,
    ResourceSpecProtocol,
)

ResourceSpecProtocolT = TypeVar("ResourceSpecProtocolT", bound=ResourceSpecProtocol)
ArtifactHyperparamsT = TypeVar("ArtifactHyperparamsT", bound=ArtifactHyperparams)
DatasetT = TypeVar("DatasetT")
ArtifactResultT = TypeVar("ArtifactResultT", bound=ArtifactResult)


@dataclass(frozen=True)
class DatasetComparisonArtifactResources(ArtifactResources, Generic[DatasetT]):
    dataset_real: DatasetT
    dataset_synthetic: DatasetT


class DatasetComparisonArtifact(
    Artifact[
        DatasetComparisonArtifactResources,
        ArtifactResultT,
        ArtifactHyperparamsT,
        ResourceSpecProtocolT,
    ],
    Generic[
        DatasetT,
        ArtifactResultT,
        ArtifactHyperparamsT,
        ResourceSpecProtocolT,
    ],
):
    @abstractmethod
    def _compare_datasets(
        self, dataset_real: DatasetT, dataset_synthetic: DatasetT
    ) -> ArtifactResultT: ...

    @abstractmethod
    def _validate_datasets(
        self, dataset_real: DatasetT, dataset_synthetic: DatasetT
    ) -> Tuple[DatasetT, DatasetT]: ...

    def _compute(self, resources: DatasetComparisonArtifactResources[DatasetT]) -> ArtifactResultT:
        result = self._compare_datasets(
            dataset_real=resources.dataset_real,
            dataset_synthetic=resources.dataset_synthetic,
        )
        return result

    def _validate(
        self, resources: DatasetComparisonArtifactResources[DatasetT]
    ) -> DatasetComparisonArtifactResources[DatasetT]:
        dataset_real_validated, dataset_synthetic_validated = self._validate_datasets(
            dataset_real=resources.dataset_real,
            dataset_synthetic=resources.dataset_synthetic,
        )
        resources_validated = DatasetComparisonArtifactResources[DatasetT](
            dataset_real=dataset_real_validated,
            dataset_synthetic=dataset_synthetic_validated,
        )
        return resources_validated
