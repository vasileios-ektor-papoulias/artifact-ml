from typing import TypeVar

from artifact_core._base.orchestration.registry import ArtifactRegistry
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
from artifact_core._domains.classification.artifact import ClassificationArtifactResources
from artifact_core._libs.resource_specs.classification.protocol import ClassSpecProtocol

ClassificationArtifactResourcesT = TypeVar(
    "ClassificationArtifactResourcesT", bound=ClassificationArtifactResources
)
ClassSpecProtocolT = TypeVar("ClassSpecProtocolT", bound=ClassSpecProtocol)
ArtifactTypeT = TypeVar("ArtifactTypeT", bound=ArtifactType)
ArtifactResultT = TypeVar("ArtifactResultT", bound=ArtifactResult)


ClassificationArtifactRegistry = ArtifactRegistry[
    ClassificationArtifactResourcesT, ClassSpecProtocolT, ArtifactTypeT, ArtifactResultT
]

ClassificationScoreRegistry = ClassificationArtifactRegistry[
    ClassificationArtifactResourcesT, ClassSpecProtocolT, ArtifactTypeT, Score
]
ClassificationArrayRegistry = ClassificationArtifactRegistry[
    ClassificationArtifactResourcesT, ClassSpecProtocolT, ArtifactTypeT, Array
]
ClassificationPlot = ClassificationArtifactRegistry[
    ClassificationArtifactResourcesT, ClassSpecProtocolT, ArtifactTypeT, Plot
]
ClassificationScoreCollectionRegistry = ClassificationArtifactRegistry[
    ClassificationArtifactResourcesT, ClassSpecProtocolT, ArtifactTypeT, ScoreCollection
]
ClassificationArrayCollectionRegistry = ClassificationArtifactRegistry[
    ClassificationArtifactResourcesT, ClassSpecProtocolT, ArtifactTypeT, ArrayCollection
]
ClassificationPlotCollectionRegistry = ClassificationArtifactRegistry[
    ClassificationArtifactResourcesT, ClassSpecProtocolT, ArtifactTypeT, PlotCollection
]
