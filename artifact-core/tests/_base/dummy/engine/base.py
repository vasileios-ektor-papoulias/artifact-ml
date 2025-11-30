from typing import TypeVar

from artifact_core._base.orchestration.engine import ArtifactEngine
from artifact_core._base.primitives.artifact_type import ArtifactType

from tests._base.dummy.resource_spec import DummyResourceSpec
from tests._base.dummy.resources import DummyArtifactResources

ScoreTypeT = TypeVar("ScoreTypeT", bound=ArtifactType)
ArrayTypeT = TypeVar("ArrayTypeT", bound=ArtifactType)
PlotTypeT = TypeVar("PlotTypeT", bound=ArtifactType)
ScoreCollectionTypeT = TypeVar("ScoreCollectionTypeT", bound=ArtifactType)
ArrayCollectionTypeT = TypeVar("ArrayCollectionTypeT", bound=ArtifactType)
PlotCollectionTypeT = TypeVar("PlotCollectionTypeT", bound=ArtifactType)

DummyArtifactEngineBase = ArtifactEngine[
    DummyArtifactResources,
    DummyResourceSpec,
    ScoreTypeT,
    ArrayTypeT,
    PlotTypeT,
    ScoreCollectionTypeT,
    ArrayCollectionTypeT,
    PlotCollectionTypeT,
]
