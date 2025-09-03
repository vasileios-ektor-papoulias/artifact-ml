from typing import TypeVar

import pandas as pd

from artifact_core.base.registry import ArtifactType
from artifact_core.core.classification.engine import ClassifierEvaluationEngine
from artifact_core.libs.resource_spec.labels.protocol import LabelsSpecProtocol

ScoreTypeT = TypeVar("ScoreTypeT", bound="ArtifactType")
ArrayTypeT = TypeVar("ArrayTypeT", bound="ArtifactType")
PlotTypeT = TypeVar("PlotTypeT", bound="ArtifactType")
ScoreCollectionTypeT = TypeVar("ScoreCollectionTypeT", bound="ArtifactType")
ArrayCollectionTypeT = TypeVar("ArrayCollectionTypeT", bound="ArtifactType")
PlotCollectionTypeT = TypeVar("PlotCollectionTypeT", bound="ArtifactType")


BinaryClassifierEvaluationEngineBase = ClassifierEvaluationEngine[
    pd.DataFrame,
    LabelsSpecProtocol,
    ScoreTypeT,
    ArrayTypeT,
    PlotTypeT,
    ScoreCollectionTypeT,
    ArrayCollectionTypeT,
    PlotCollectionTypeT,
]
