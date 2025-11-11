from typing import TypeVar

from artifact_core._base.artifact_dependencies import (
    ArtifactResult,
)
from artifact_core._base.registry import ArtifactRegistry, ArtifactType
from artifact_core._core.classification.artifact import ClassificationArtifactResources
from artifact_core._libs.resource_spec.categorical.protocol import CategoricalFeatureSpecProtocol

ClassificationArtifactResourcesT = TypeVar(
    "ClassificationArtifactResourcesT", bound=ClassificationArtifactResources
)
CategoricalFeatureSpecProtocolT = TypeVar(
    "CategoricalFeatureSpecProtocolT", bound=CategoricalFeatureSpecProtocol
)
ArtifactTypeT = TypeVar("ArtifactTypeT", bound=ArtifactType)
ArtifactResultT = TypeVar("ArtifactResultT", bound=ArtifactResult)


ClassificationArtifactRegistry = ArtifactRegistry[
    ClassificationArtifactResourcesT,
    CategoricalFeatureSpecProtocolT,
    ArtifactTypeT,
    ArtifactResultT,
]
