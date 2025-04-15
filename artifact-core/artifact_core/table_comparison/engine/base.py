from typing import TypeVar

import pandas as pd

from artifact_core.base.registry import ArtifactType
from artifact_core.core.dataset_comparison.engine import (
    DatasetComparisonEngine,
)
from artifact_core.libs.data_spec.tabular.protocol import (
    TabularDataSpecProtocol,
)

scoreTypeT = TypeVar("scoreTypeT", bound="ArtifactType")
arrayTypeT = TypeVar("arrayTypeT", bound="ArtifactType")
plotTypeT = TypeVar("plotTypeT", bound="ArtifactType")
scoreCollectionTypeT = TypeVar("scoreCollectionTypeT", bound="ArtifactType")
arrayCollectionTypeT = TypeVar("arrayCollectionTypeT", bound="ArtifactType")
plotCollectionTypeT = TypeVar("plotCollectionTypeT", bound="ArtifactType")


TableComparisonEngineBase = DatasetComparisonEngine[
    pd.DataFrame,
    TabularDataSpecProtocol,
    scoreTypeT,
    arrayTypeT,
    plotTypeT,
    scoreCollectionTypeT,
    arrayCollectionTypeT,
    plotCollectionTypeT,
]
