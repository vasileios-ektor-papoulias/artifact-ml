from typing import TypeVar

from artifact_core._base.orchestration.registry import ArtifactRegistry
from artifact_core._base.types.artifact_result import ArtifactResult
from artifact_core._base.types.artifact_type import ArtifactType
from artifact_core._libs.resource_specs.classification.protocol import (
    CategoricalFeatureSpecProtocol,
)
from artifact_core._tasks.classification.artifact import ClassificationArtifactResources

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
