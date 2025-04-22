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

artifactTypeT = TypeVar("artifactTypeT", bound="ArtifactType")
artifactResultT = TypeVar("artifactResultT", bound=ArtifactResult)


TableComparisonArtifactRegistry = DatasetComparisonArtifactRegistry[
    artifactTypeT,
    pd.DataFrame,
    artifactResultT,
    TabularDataSpecProtocol,
]

TableComparisonScoreRegistryBase = TableComparisonArtifactRegistry[artifactTypeT, float]
TableComparisonArrayRegistryBase = TableComparisonArtifactRegistry[artifactTypeT, ndarray]
TableComparisonPlotRegistryBase = TableComparisonArtifactRegistry[artifactTypeT, Figure]
TableComparisonScoreCollectionRegistryBase = TableComparisonArtifactRegistry[
    artifactTypeT, Dict[str, float]
]
TableComparisonArrayCollectionRegistryBase = TableComparisonArtifactRegistry[
    artifactTypeT, Dict[str, ndarray]
]
TableComparisonPlotCollectionRegistryBase = TableComparisonArtifactRegistry[
    artifactTypeT, Dict[str, Figure]
]
