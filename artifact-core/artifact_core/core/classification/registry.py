from typing import TypeVar

from artifact_core.base.artifact_dependencies import (
    ArtifactResult,
)
from artifact_core.base.registry import ArtifactRegistry, ArtifactType
from artifact_core.core.classification.artifact import ClassificationArtifactResources
from artifact_core.libs.resource_spec.categorical.protocol import CategoricalFeatureSpecProtocol

ArtifactTypeT = TypeVar("ArtifactTypeT", bound="ArtifactType")
ClassificationArtifactResourcesT = TypeVar(
    "ClassificationArtifactResourcesT", bound=ClassificationArtifactResources
)
ArtifactResultT = TypeVar("ArtifactResultT", bound=ArtifactResult)
CategoricalFeatureSpecProtocolT = TypeVar(
    "CategoricalFeatureSpecProtocolT", bound=CategoricalFeatureSpecProtocol
)


ClassificationArtifactRegistry = ArtifactRegistry[
    ArtifactTypeT,
    ClassificationArtifactResourcesT,
    ArtifactResultT,
    CategoricalFeatureSpecProtocolT,
]
