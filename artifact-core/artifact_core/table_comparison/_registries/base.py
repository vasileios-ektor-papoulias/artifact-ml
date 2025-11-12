from typing import Dict, TypeVar

import pandas as pd
from matplotlib.figure import Figure

from artifact_core._base.types.artifact_result import Array, ArtifactResult
from artifact_core._base.types.artifact_type import ArtifactType
from artifact_core._libs.resource_specs.table_comparison.protocol import TabularDataSpecProtocol
from artifact_core._tasks.dataset_comparison.registry import DatasetComparisonArtifactRegistry

ArtifactTypeT = TypeVar("ArtifactTypeT", bound=ArtifactType)
ArtifactResultT = TypeVar("ArtifactResultT", bound=ArtifactResult)


TableComparisonArtifactRegistry = DatasetComparisonArtifactRegistry[
    pd.DataFrame, TabularDataSpecProtocol, ArtifactTypeT, ArtifactResultT
]

TableComparisonScoreRegistryBase = TableComparisonArtifactRegistry[ArtifactTypeT, float]
TableComparisonArrayRegistryBase = TableComparisonArtifactRegistry[ArtifactTypeT, Array]
TableComparisonPlotRegistryBase = TableComparisonArtifactRegistry[ArtifactTypeT, Figure]
TableComparisonScoreCollectionRegistryBase = TableComparisonArtifactRegistry[
    ArtifactTypeT, Dict[str, float]
]
TableComparisonArrayCollectionRegistryBase = TableComparisonArtifactRegistry[
    ArtifactTypeT, Dict[str, Array]
]
TableComparisonPlotCollectionRegistryBase = TableComparisonArtifactRegistry[
    ArtifactTypeT, Dict[str, Figure]
]
