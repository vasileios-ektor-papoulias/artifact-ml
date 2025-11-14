from typing import TypeVar

import pandas as pd

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
from artifact_core._domains.dataset_comparison.registry import DatasetComparisonArtifactRegistry
from artifact_core._libs.resource_specs.table_comparison.protocol import TabularDataSpecProtocol

ArtifactTypeT = TypeVar("ArtifactTypeT", bound=ArtifactType)
ArtifactResultT = TypeVar("ArtifactResultT", bound=ArtifactResult)


TableComparisonArtifactRegistry = DatasetComparisonArtifactRegistry[
    pd.DataFrame, TabularDataSpecProtocol, ArtifactTypeT, ArtifactResultT
]

TableComparisonScoreRegistryBase = TableComparisonArtifactRegistry[ArtifactTypeT, Score]
TableComparisonArrayRegistryBase = TableComparisonArtifactRegistry[ArtifactTypeT, Array]
TableComparisonPlotRegistryBase = TableComparisonArtifactRegistry[ArtifactTypeT, Plot]
TableComparisonScoreCollectionRegistryBase = TableComparisonArtifactRegistry[
    ArtifactTypeT, ScoreCollection
]
TableComparisonArrayCollectionRegistryBase = TableComparisonArtifactRegistry[
    ArtifactTypeT, ArrayCollection
]
TableComparisonPlotCollectionRegistryBase = TableComparisonArtifactRegistry[
    ArtifactTypeT, PlotCollection
]
