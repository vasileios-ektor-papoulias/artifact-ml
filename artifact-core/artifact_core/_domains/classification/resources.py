from dataclasses import dataclass
from typing import Generic, TypeVar

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
