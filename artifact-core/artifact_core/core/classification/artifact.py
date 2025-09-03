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
ArtifactHyperparamsT = TypeVar("ArtifactHyperparamsT", bound="ArtifactHyperparams")
LabelsT = TypeVar("LabelsT")
ArtifactResultT = TypeVar("ArtifactResultT", bound=ArtifactResult)


@dataclass(frozen=True)
class ClassificationArtifactResources(ArtifactResources, Generic[LabelsT]):
    labels_ground_truth: LabelsT
    labels_predicted: LabelsT


class ClassificationArtifact(
    Artifact[
        ClassificationArtifactResources,
        ArtifactResultT,
        ArtifactHyperparamsT,
        ResourceSpecProtocolT,
    ],
    Generic[
        LabelsT,
        ArtifactResultT,
        ArtifactHyperparamsT,
        ResourceSpecProtocolT,
    ],
):
    @abstractmethod
    def _evaluate_classification(
        self, labels_ground_truth: LabelsT, labels_predicted: LabelsT
    ) -> ArtifactResultT: ...

    @abstractmethod
    def _validate_labels(
        self, labels_ground_truth: LabelsT, labels_predicted: LabelsT
    ) -> Tuple[LabelsT, LabelsT]: ...

    def _compute(self, resources: ClassificationArtifactResources[LabelsT]) -> ArtifactResultT:
        result = self._evaluate_classification(
            labels_ground_truth=resources.labels_ground_truth,
            labels_predicted=resources.labels_predicted,
        )
        return result

    def _validate(
        self, resources: ClassificationArtifactResources[LabelsT]
    ) -> ClassificationArtifactResources[LabelsT]:
        labels_ground_truth_validated, labels_predicted_validated = self._validate_labels(
            labels_ground_truth=resources.labels_ground_truth,
            labels_predicted=resources.labels_predicted,
        )
        resources_validated = ClassificationArtifactResources[LabelsT](
            labels_ground_truth=labels_ground_truth_validated,
            labels_predicted=labels_predicted_validated,
        )
        return resources_validated
