from typing import TypeVar

from artifact_core._tasks.classification.artifact import ClassificationArtifactResources

from artifact_experiment.base.components.callbacks.artifact import ArtifactCallbackResources

ClassificationArtifactResourcesTCov = TypeVar(
    "ClassificationArtifactResourcesTCov", bound=ClassificationArtifactResources, covariant=True
)


ClassificationCallbackResources = ArtifactCallbackResources[ClassificationArtifactResourcesTCov]
