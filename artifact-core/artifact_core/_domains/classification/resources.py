from dataclasses import dataclass
from typing import Any, Generic, Mapping, TypeVar

from artifact_core._base.core.resources import ArtifactResources
from artifact_core._libs.resources.classification.class_store import ClassStore
from artifact_core._libs.resources.classification.classification_results import (
    ClassificationResults,
)

CategoryStoreT = TypeVar("CategoryStoreT", bound=ClassStore)
ClassificationResultsT = TypeVar("ClassificationResultsT", bound=ClassificationResults)


@dataclass(frozen=True)
class ClassificationArtifactResources(
    ArtifactResources, Generic[CategoryStoreT, ClassificationResultsT]
):
    true_class_store: CategoryStoreT
    classification_results: ClassificationResultsT

    def serialize(self) -> Mapping[str, Any]:
        true = self.true_class_store.id_to_class_name
        true = {str(identifier): category for identifier, category in true.items()}
        predicted = self.classification_results.id_to_predicted_class
        predicted = {str(identifier): category for identifier, category in predicted.items()}
        dict_artifact_resources = {
            identifier: {"true": true.get(identifier), "predicted": predicted.get(identifier)}
            for identifier in set(true) | set(predicted)
        }
        return dict_artifact_resources
