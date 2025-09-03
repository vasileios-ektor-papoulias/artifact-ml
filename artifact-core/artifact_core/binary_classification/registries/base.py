from typing import Dict, TypeVar

import pandas as pd
from matplotlib.figure import Figure
from numpy import ndarray

from artifact_core.base.artifact_dependencies import ArtifactResult
from artifact_core.base.registry import ArtifactType
from artifact_core.core.dataset_comparison.registry import (
    DatasetComparisonArtifactRegistry,
)
from artifact_core.libs.resource_spec.labels.protocol import LabelsSpecProtocol

ArtifactTypeT = TypeVar("ArtifactTypeT", bound="ArtifactType")
ArtifactResultT = TypeVar("ArtifactResultT", bound=ArtifactResult)


BinaryClassificationArtifactRegistry = DatasetComparisonArtifactRegistry[
    ArtifactTypeT,
    pd.DataFrame,
    ArtifactResultT,
    LabelsSpecProtocol,
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
