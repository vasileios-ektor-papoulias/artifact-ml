from typing import Generic, TypeVar

from artifact_core.base.artifact_dependencies import (
    ArtifactResult,
)
from artifact_core.base.registry import ArtifactRegistry, ArtifactType
from artifact_core.core.classification.artifact import ClassificationArtifactResources
from artifact_core.libs.resource_spec.categorical.protocol import CategoricalFeatureSpecProtocol

ArtifactTypeT = TypeVar("ArtifactTypeT", bound="ArtifactType")
ArtifactResultT = TypeVar("ArtifactResultT", bound=ArtifactResult)
CategoricalFeatureSpecProtocolT = TypeVar(
    "CategoricalFeatureSpecProtocolT", bound=CategoricalFeatureSpecProtocol
)


class ClassificationArtifactRegistry(
    ArtifactRegistry[
        ArtifactTypeT,
        ClassificationArtifactResources[CategoricalFeatureSpecProtocolT],
        ArtifactResultT,
        CategoricalFeatureSpecProtocolT,
    ],
    Generic[ArtifactTypeT, ArtifactResultT, CategoricalFeatureSpecProtocolT],
):
    pass
