from typing import TypeVar

from artifact_core.core.classification.artifact import ClassificationArtifactResources

from artifact_experiment.base.callbacks.artifact import ArtifactCallbackResources

ClassificationArtifactResourcesTCov = TypeVar(
    "ClassificationArtifactResourcesTCov", bound=ClassificationArtifactResources, covariant=True
)


ClassificationCallbackResources = ArtifactCallbackResources[ClassificationArtifactResourcesTCov]
