from typing import TypeVar

from artifact_core.core.classification.artifact import ClassificationArtifactResources

from artifact_experiment.base.callbacks.artifact import ArtifactCallbackResources

ClassificationArtifactResourcesT = TypeVar(
    "ClassificationArtifactResourcesT", bound=ClassificationArtifactResources
)


ClassificationCallbackResources = ArtifactCallbackResources[ClassificationArtifactResourcesT]
