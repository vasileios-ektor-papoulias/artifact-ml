from typing import TypeVar

from artifact_core.spi.resources import ClassificationArtifactResources

from artifact_experiment._base.components.callbacks.artifact import ArtifactCallbackResources

ClassificationArtifactResourcesTCov = TypeVar(
    "ClassificationArtifactResourcesTCov", bound=ClassificationArtifactResources, covariant=True
)


ClassificationCallbackResources = ArtifactCallbackResources[ClassificationArtifactResourcesTCov]
