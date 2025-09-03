from typing import Dict, TypeVar

import pandas as pd
from matplotlib.figure import Figure
from numpy import ndarray

from artifact_core.base.artifact_dependencies import ArtifactResult
from artifact_core.base.registry import ArtifactType
from artifact_core.core.classification.registry import (
    ClassificationArtifactRegistry,
)
from artifact_core.libs.resource_spec.labels.protocol import LabelSpecProtocol

ArtifactTypeT = TypeVar("ArtifactTypeT", bound="ArtifactType")
ArtifactResultT = TypeVar("ArtifactResultT", bound=ArtifactResult)


BinaryClassificationArtifactRegistry = ClassificationArtifactRegistry[
    ArtifactTypeT,
    pd.DataFrame,
    ArtifactResultT,
    LabelSpecProtocol,
]

BinaryClassificationScoreRegistryBase = BinaryClassificationArtifactRegistry[ArtifactTypeT, float]
BinaryClassificationArrayRegistryBase = BinaryClassificationArtifactRegistry[ArtifactTypeT, ndarray]
BinaryClassificationPlotRegistryBase = BinaryClassificationArtifactRegistry[ArtifactTypeT, Figure]
BinaryClassificationScoreCollectionRegistryBase = BinaryClassificationArtifactRegistry[
    ArtifactTypeT, Dict[str, float]
]
BinaryClassificationArrayCollectionRegistryBase = BinaryClassificationArtifactRegistry[
    ArtifactTypeT, Dict[str, ndarray]
]
BinaryClassificationPlotCollectionRegistryBase = BinaryClassificationArtifactRegistry[
    ArtifactTypeT, Dict[str, Figure]
]
