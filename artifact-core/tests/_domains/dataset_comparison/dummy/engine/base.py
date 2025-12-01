from typing import TypeVar

from artifact_core._domains.dataset_comparison.engine import DatasetComparisonEngine

from tests._domains.dataset_comparison.dummy.resource_spec import DummyDatasetSpec
from tests._domains.dataset_comparison.dummy.resources import DummyDataset

ScoreTypeT = TypeVar("ScoreTypeT")
ArrayTypeT = TypeVar("ArrayTypeT")
PlotTypeT = TypeVar("PlotTypeT")
ScoreCollectionTypeT = TypeVar("ScoreCollectionTypeT")
ArrayCollectionTypeT = TypeVar("ArrayCollectionTypeT")
PlotCollectionTypeT = TypeVar("PlotCollectionTypeT")

DummyDatasetComparisonEngineBase = DatasetComparisonEngine[
    DummyDataset,
    DummyDatasetSpec,
    ScoreTypeT,
    ArrayTypeT,
    PlotTypeT,
    ScoreCollectionTypeT,
    ArrayCollectionTypeT,
    PlotCollectionTypeT,
]
