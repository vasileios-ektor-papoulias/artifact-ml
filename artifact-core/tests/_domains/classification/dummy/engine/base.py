from typing import TypeVar

from artifact_core._base.primitives.artifact_type import ArtifactType
from artifact_core._domains.classification.engine import ClassificationEngine

from tests._domains.classification.dummy.resource_spec import DummyClassSpec
from tests._domains.classification.dummy.resources import (
    DummyClassificationResults,
    DummyClassStore,
)

ScoreTypeT = TypeVar("ScoreTypeT", bound=ArtifactType)
ArrayTypeT = TypeVar("ArrayTypeT", bound=ArtifactType)
PlotTypeT = TypeVar("PlotTypeT", bound=ArtifactType)
ScoreCollectionTypeT = TypeVar("ScoreCollectionTypeT", bound=ArtifactType)
ArrayCollectionTypeT = TypeVar("ArrayCollectionTypeT", bound=ArtifactType)
PlotCollectionTypeT = TypeVar("PlotCollectionTypeT", bound=ArtifactType)


DummyClassificationEngineBase = ClassificationEngine[
    DummyClassStore,
    DummyClassificationResults,
    DummyClassSpec,
    ScoreTypeT,
    ArrayTypeT,
    PlotTypeT,
    ScoreCollectionTypeT,
    ArrayCollectionTypeT,
    PlotCollectionTypeT,
]
