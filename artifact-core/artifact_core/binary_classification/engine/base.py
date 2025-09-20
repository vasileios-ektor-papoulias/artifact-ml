from typing import TypeVar

from artifact_core.base.registry import ArtifactType
from artifact_core.core.classification.engine import ClassifierEvaluationEngine
from artifact_core.libs.resource_spec.binary.protocol import BinaryFeatureSpecProtocol

ScoreTypeT = TypeVar("ScoreTypeT", bound="ArtifactType")
ArrayTypeT = TypeVar("ArrayTypeT", bound="ArtifactType")
PlotTypeT = TypeVar("PlotTypeT", bound="ArtifactType")
ScoreCollectionTypeT = TypeVar("ScoreCollectionTypeT", bound="ArtifactType")
ArrayCollectionTypeT = TypeVar("ArrayCollectionTypeT", bound="ArtifactType")
PlotCollectionTypeT = TypeVar("PlotCollectionTypeT", bound="ArtifactType")


BinaryClassifierEvaluationEngineBase = ClassifierEvaluationEngine[
    BinaryFeatureSpecProtocol,
    ScoreTypeT,
    ArrayTypeT,
    PlotTypeT,
    ScoreCollectionTypeT,
    ArrayCollectionTypeT,
    PlotCollectionTypeT,
]
