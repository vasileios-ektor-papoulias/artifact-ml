from typing import TypeVar

from artifact_core._base.primitives.artifact_type import ArtifactType
from artifact_core._base.typing.artifact_result import ArtifactResult
from artifact_core._domains.classification.registry import (
    ClassificationArtifactRegistry,
)

from tests._domains.classification.dummy.resource_spec import DummyClassSpec
from tests._domains.classification.dummy.resources import (
    DummyClassificationResults,
    DummyClassStore,
)

ArtifactTypeT = TypeVar("ArtifactTypeT", bound=ArtifactType)
ArtifactResultT = TypeVar("ArtifactResultT", bound=ArtifactResult)

DummyClassificationRegistry = ClassificationArtifactRegistry[
    DummyClassStore,
    DummyClassificationResults,
    DummyClassSpec,
    ArtifactTypeT,
    ArtifactResultT,
]
