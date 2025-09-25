from typing import Dict, TypeVar

import pandas as pd
from matplotlib.figure import Figure
from numpy import ndarray

from artifact_core.base.artifact_dependencies import ArtifactResult
from artifact_core.base.registry import ArtifactType
from artifact_core.core.dataset_comparison.registry import (
    DatasetComparisonArtifactRegistry,
)
from artifact_core.libs.resource_spec.tabular.protocol import (
    TabularDataSpecProtocol,
)

ArtifactTypeT = TypeVar("ArtifactTypeT", bound=ArtifactType)
ArtifactResultT = TypeVar("ArtifactResultT", bound=ArtifactResult)


TableComparisonArtifactRegistry = DatasetComparisonArtifactRegistry[
    ArtifactTypeT,
    pd.DataFrame,
    ArtifactResultT,
    TabularDataSpecProtocol,
]

TableComparisonScoreRegistryBase = TableComparisonArtifactRegistry[ArtifactTypeT, float]
TableComparisonArrayRegistryBase = TableComparisonArtifactRegistry[ArtifactTypeT, ndarray]
TableComparisonPlotRegistryBase = TableComparisonArtifactRegistry[ArtifactTypeT, Figure]
TableComparisonScoreCollectionRegistryBase = TableComparisonArtifactRegistry[
    ArtifactTypeT, Dict[str, float]
]
TableComparisonArrayCollectionRegistryBase = TableComparisonArtifactRegistry[
    ArtifactTypeT, Dict[str, ndarray]
]
TableComparisonPlotCollectionRegistryBase = TableComparisonArtifactRegistry[
    ArtifactTypeT, Dict[str, Figure]
]
