from typing import Dict, TypeVar

from matplotlib.figure import Figure
from numpy import ndarray

from artifact_core._base.artifact_dependencies import ArtifactResult
from artifact_core._base.registry import ArtifactType
from artifact_core._core.classification.registry import ClassificationArtifactRegistry
from artifact_core._libs.resource_spec.binary.protocol import BinaryFeatureSpecProtocol
from artifact_core.binary_classification._artifacts.base import (
    BinaryClassificationArtifactResources,
)

ArtifactTypeT = TypeVar("ArtifactTypeT", bound=ArtifactType)
ArtifactResultT = TypeVar("ArtifactResultT", bound=ArtifactResult)


BinaryClassificationArtifactRegistry = ClassificationArtifactRegistry[
    BinaryClassificationArtifactResources, BinaryFeatureSpecProtocol, ArtifactTypeT, ArtifactResultT
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
