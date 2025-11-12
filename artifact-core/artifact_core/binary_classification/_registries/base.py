from typing import Dict, TypeVar

from matplotlib.figure import Figure

from artifact_core._base.types.artifact_result import Array, ArtifactResult
from artifact_core._base.types.artifact_type import ArtifactType
from artifact_core._libs.resource_specs.binary_classification.protocol import (
    BinaryFeatureSpecProtocol,
)
from artifact_core._tasks.classification.registry import ClassificationArtifactRegistry
from artifact_core.binary_classification._artifacts.base import (
    BinaryClassificationArtifactResources,
)

ArtifactTypeT = TypeVar("ArtifactTypeT", bound=ArtifactType)
ArtifactResultT = TypeVar("ArtifactResultT", bound=ArtifactResult)


BinaryClassificationArtifactRegistry = ClassificationArtifactRegistry[
    BinaryClassificationArtifactResources, BinaryFeatureSpecProtocol, ArtifactTypeT, ArtifactResultT
]

BinaryClassificationScoreRegistryBase = BinaryClassificationArtifactRegistry[ArtifactTypeT, float]
BinaryClassificationArrayRegistryBase = BinaryClassificationArtifactRegistry[ArtifactTypeT, Array]
BinaryClassificationPlotRegistryBase = BinaryClassificationArtifactRegistry[ArtifactTypeT, Figure]
BinaryClassificationScoreCollectionRegistryBase = BinaryClassificationArtifactRegistry[
    ArtifactTypeT, Dict[str, float]
]
BinaryClassificationArrayCollectionRegistryBase = BinaryClassificationArtifactRegistry[
    ArtifactTypeT, Dict[str, Array]
]
BinaryClassificationPlotCollectionRegistryBase = BinaryClassificationArtifactRegistry[
    ArtifactTypeT, Dict[str, Figure]
]
