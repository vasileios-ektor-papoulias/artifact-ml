from typing import TypeVar

import pandas as pd

from artifact_core._base.primitives.artifact_type import ArtifactType
from artifact_core._domains.dataset_comparison.engine import DatasetComparisonEngine
from artifact_core._libs.resource_specs.table_comparison.protocol import TabularDataSpecProtocol

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
