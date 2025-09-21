from typing import TypeVar

from artifact_core.base.engine import ArtifactEngine
from artifact_core.base.registry import ArtifactType
from artifact_core.core.classification.artifact import ClassificationArtifactResources
from artifact_core.libs.resource_spec.categorical.protocol import CategoricalFeatureSpecProtocol

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


ClassifierEvaluationEngine = ArtifactEngine[
    ClassificationArtifactResourcesT,
    CategoricalFeatureSpecProtocolT,
    ScoreTypeT,
    ArrayTypeT,
    PlotTypeT,
    ScoreCollectionTypeT,
    ArrayCollectionTypeT,
    PlotCollectionTypeT,
]
