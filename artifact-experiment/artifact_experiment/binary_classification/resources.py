from typing import Type, TypeVar

from artifact_core.binary_classification.artifacts.base import BinaryClassificationArtifactResources
from artifact_core.libs.resources.classification.classification_results import (
    BinaryClassificationResults,
)

from artifact_experiment.base.callbacks.artifact import ArtifactCallbackResources

BinaryClassificationCallbackResourcesT = TypeVar(
    "BinaryClassificationCallbackResourcesT", bound="BinaryClassificationCallbackResources"
)


class BinaryClassificationCallbackResources(
    ArtifactCallbackResources[BinaryClassificationArtifactResources]
):
    @classmethod
    def build(
        cls: Type[BinaryClassificationCallbackResourcesT],
        classification_results: BinaryClassificationResults,
    ) -> BinaryClassificationCallbackResourcesT:
        artifact_resources = BinaryClassificationArtifactResources(
            classification_results=classification_results
        )
        callback_resources = cls(artifact_resources=artifact_resources)
        return callback_resources
