from typing import TypeVar

from artifact_core._base.primitives.artifact_type import ArtifactType
from artifact_core._base.typing.artifact_result import (
    Array,
    ArrayCollection,
    ArtifactResult,
    Plot,
    PlotCollection,
    Score,
    ScoreCollection,
)
from artifact_core._domains.classification.registry import ClassificationArtifactRegistry
from artifact_core._libs.resource_specs.binary_classification.protocol import (
    BinaryClassSpecProtocol,
)
from artifact_core.binary_classification._resources import BinaryClassificationArtifactResources

ArtifactTypeT = TypeVar("ArtifactTypeT", bound=ArtifactType)
ArtifactResultT = TypeVar("ArtifactResultT", bound=ArtifactResult)


BinaryClassificationArtifactRegistry = ClassificationArtifactRegistry[
    BinaryClassificationArtifactResources, BinaryClassSpecProtocol, ArtifactTypeT, ArtifactResultT
]

BinaryClassificationScoreRegistryBase = BinaryClassificationArtifactRegistry[ArtifactTypeT, Score]
BinaryClassificationArrayRegistryBase = BinaryClassificationArtifactRegistry[ArtifactTypeT, Array]
BinaryClassificationPlotRegistryBase = BinaryClassificationArtifactRegistry[ArtifactTypeT, Plot]
BinaryClassificationScoreCollectionRegistryBase = BinaryClassificationArtifactRegistry[
    ArtifactTypeT, ScoreCollection
]
BinaryClassificationArrayCollectionRegistryBase = BinaryClassificationArtifactRegistry[
    ArtifactTypeT, ArrayCollection
]
BinaryClassificationPlotCollectionRegistryBase = BinaryClassificationArtifactRegistry[
    ArtifactTypeT, PlotCollection
]
