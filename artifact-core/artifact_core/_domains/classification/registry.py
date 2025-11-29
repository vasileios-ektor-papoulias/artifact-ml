from typing import TypeVar

from artifact_core._base.orchestration.registry import ArtifactRegistry
from artifact_core._base.primitives.artifact_type import ArtifactType
from artifact_core._base.typing.artifact_result import ArtifactResult
from artifact_core._domains.classification.artifact import ClassificationArtifactResources
from artifact_core._libs.resource_specs.classification.protocol import ClassSpecProtocol

ClassificationArtifactResourcesT = TypeVar(
    "ClassificationArtifactResourcesT", bound=ClassificationArtifactResources
)
ClassSpecProtocolT = TypeVar("ClassSpecProtocolT", bound=ClassSpecProtocol)
ArtifactTypeT = TypeVar("ArtifactTypeT", bound=ArtifactType)
ArtifactResultT = TypeVar("ArtifactResultT", bound=ArtifactResult)


ClassificationArtifactRegistry = ArtifactRegistry[
    ClassificationArtifactResourcesT, ClassSpecProtocolT, ArtifactTypeT, ArtifactResultT
]
