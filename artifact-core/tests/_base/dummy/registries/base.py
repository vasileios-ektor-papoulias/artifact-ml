from typing import TypeVar

from artifact_core._base.orchestration.registry import (
    ArtifactRegistry,
)
from artifact_core._base.primitives.artifact_type import ArtifactType
from artifact_core._base.typing.artifact_result import ArtifactResult

from tests._base.dummy.resource_spec import DummyResourceSpec
from tests._base.dummy.resources import DummyArtifactResources

ArtifactTypeT = TypeVar("ArtifactTypeT", bound=ArtifactType)
ArtifactResultT = TypeVar("ArtifactResultT", bound=ArtifactResult)


DummyArtifactRegistry = ArtifactRegistry[
    DummyArtifactResources, DummyResourceSpec, ArtifactTypeT, ArtifactResultT
]
