from typing import TypeVar

from artifact_core._base.orchestration.registry import ArtifactRegistry
from artifact_core._base.primitives.artifact_type import ArtifactType
from artifact_core._base.typing.artifact_result import ArtifactResult
from artifact_core._domains.classification.artifact import ClassificationArtifactResources
from artifact_core._libs.resource_specs.classification.protocol import ClassSpecProtocol
from artifact_core._libs.resources.classification.class_store import ClassStore
from artifact_core._libs.resources.classification.classification_results import (
    ClassificationResults,
)

ClassStoreT = TypeVar("ClassStoreT", bound=ClassStore)
ClassificationResultsT = TypeVar("ClassificationResultsT", bound=ClassificationResults)
ClassSpecProtocolT = TypeVar("ClassSpecProtocolT", bound=ClassSpecProtocol)
ArtifactTypeT = TypeVar("ArtifactTypeT", bound=ArtifactType)
ArtifactResultT = TypeVar("ArtifactResultT", bound=ArtifactResult)


ClassificationArtifactRegistry = ArtifactRegistry[
    ClassificationArtifactResources[
        ClassStoreT,
        ClassificationResultsT,
    ],
    ClassSpecProtocolT,
    ArtifactTypeT,
    ArtifactResultT,
]
