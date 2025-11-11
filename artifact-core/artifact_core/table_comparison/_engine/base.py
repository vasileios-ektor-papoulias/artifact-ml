from typing import TypeVar

import pandas as pd

from artifact_core._base.registry import ArtifactType
from artifact_core._core.dataset_comparison.engine import (
    DatasetComparisonEngine,
)
from artifact_core._libs.resource_spec.tabular.protocol import (
    TabularDataSpecProtocol,
)

ScoreTypeT = TypeVar("ScoreTypeT", bound=ArtifactType)
ArrayTypeT = TypeVar("ArrayTypeT", bound=ArtifactType)
PlotTypeT = TypeVar("PlotTypeT", bound=ArtifactType)
ScoreCollectionTypeT = TypeVar("ScoreCollectionTypeT", bound=ArtifactType)
ArrayCollectionTypeT = TypeVar("ArrayCollectionTypeT", bound=ArtifactType)
PlotCollectionTypeT = TypeVar("PlotCollectionTypeT", bound=ArtifactType)


TableComparisonEngineBase = DatasetComparisonEngine[
    pd.DataFrame,
    TabularDataSpecProtocol,
    ScoreTypeT,
    ArrayTypeT,
    PlotTypeT,
    ScoreCollectionTypeT,
    ArrayCollectionTypeT,
    PlotCollectionTypeT,
]
