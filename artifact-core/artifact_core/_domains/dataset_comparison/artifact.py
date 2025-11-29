from abc import abstractmethod
from typing import Generic, Tuple, TypeVar

from artifact_core._base.core.artifact import Artifact
from artifact_core._base.core.hyperparams import ArtifactHyperparams
from artifact_core._base.core.resource_spec import ResourceSpecProtocol
from artifact_core._base.typing.artifact_result import ArtifactResult
from artifact_core._domains.dataset_comparison.resources import DatasetComparisonArtifactResources

DatasetT = TypeVar("DatasetT")
ResourceSpecProtocolT = TypeVar("ResourceSpecProtocolT", bound=ResourceSpecProtocol)
ArtifactHyperparamsT = TypeVar("ArtifactHyperparamsT", bound=ArtifactHyperparams)
ArtifactResultT = TypeVar("ArtifactResultT", bound=ArtifactResult)


class DatasetComparisonArtifact(
    Artifact[
        DatasetComparisonArtifactResources[DatasetT],
        ResourceSpecProtocolT,
        ArtifactHyperparamsT,
        ArtifactResultT,
    ],
    Generic[DatasetT, ResourceSpecProtocolT, ArtifactHyperparamsT, ArtifactResultT],
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
