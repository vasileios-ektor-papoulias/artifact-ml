from typing import TypeVar

from artifact_core._base.orchestration.engine import ArtifactEngine
from artifact_core._base.orchestration.registry import ArtifactType
from artifact_core._libs.resource_specs.classification.protocol import (
    CategoricalFeatureSpecProtocol,
)
from artifact_core._tasks.classification.artifact import ClassificationArtifactResources

ScoreTypeT = TypeVar("ScoreTypeT", bound=ArtifactType)
ArrayTypeT = TypeVar("ArrayTypeT", bound=ArtifactType)
PlotTypeT = TypeVar("PlotTypeT", bound=ArtifactType)
ScoreCollectionTypeT = TypeVar("ScoreCollectionTypeT", bound=ArtifactType)
ArrayCollectionTypeT = TypeVar("ArrayCollectionTypeT", bound=ArtifactType)
PlotCollectionTypeT = TypeVar("PlotCollectionTypeT", bound=ArtifactType)
ClassificationArtifactResourcesT = TypeVar(
    "ClassificationArtifactResourcesT", bound=ClassificationArtifactResources
)
CategoricalFeatureSpecProtocolT = TypeVar(
    "CategoricalFeatureSpecProtocolT", bound=CategoricalFeatureSpecProtocol
)


ClassificationEngine = ArtifactEngine[
    ClassificationArtifactResourcesT,
    CategoricalFeatureSpecProtocolT,
    ScoreTypeT,
    ArrayTypeT,
    PlotTypeT,
    ScoreCollectionTypeT,
    ArrayCollectionTypeT,
    PlotCollectionTypeT,
]
