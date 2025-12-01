from typing import Type

from tests._domains.dataset_comparison.dummy.engine.base import DummyDatasetComparisonEngineBase
from tests._domains.dataset_comparison.dummy.registries.array_collections import (
    DummyDatasetComparisonArrayCollectionRegistry,
)
from tests._domains.dataset_comparison.dummy.registries.arrays import (
    DummyDatasetComparisonArrayRegistry,
)
from tests._domains.dataset_comparison.dummy.registries.plot_collections import (
    DummyDatasetComparisonPlotCollectionRegistry,
)
from tests._domains.dataset_comparison.dummy.registries.plots import (
    DummyDatasetComparisonPlotRegistry,
)
from tests._domains.dataset_comparison.dummy.registries.score_collections import (
    DummyDatasetComparisonScoreCollectionRegistry,
)
from tests._domains.dataset_comparison.dummy.registries.scores import (
    DummyDatasetComparisonScoreRegistry,
)
from tests._domains.dataset_comparison.dummy.types.array_collections import (
    DummyDatasetComparisonArrayCollectionType,
)
from tests._domains.dataset_comparison.dummy.types.arrays import DummyDatasetComparisonArrayType
from tests._domains.dataset_comparison.dummy.types.plot_collections import (
    DummyDatasetComparisonPlotCollectionType,
)
from tests._domains.dataset_comparison.dummy.types.plots import DummyDatasetComparisonPlotType
from tests._domains.dataset_comparison.dummy.types.score_collections import (
    DummyDatasetComparisonScoreCollectionType,
)
from tests._domains.dataset_comparison.dummy.types.scores import DummyDatasetComparisonScoreType


class DummyDatasetComparisonEngine(
    DummyDatasetComparisonEngineBase[
        DummyDatasetComparisonScoreType,
        DummyDatasetComparisonArrayType,
        DummyDatasetComparisonPlotType,
        DummyDatasetComparisonScoreCollectionType,
        DummyDatasetComparisonArrayCollectionType,
        DummyDatasetComparisonPlotCollectionType,
    ]
):
    @classmethod
    def _get_score_registry(cls) -> Type[DummyDatasetComparisonScoreRegistry]:
        return DummyDatasetComparisonScoreRegistry

    @classmethod
    def _get_array_registry(cls) -> Type[DummyDatasetComparisonArrayRegistry]:
        return DummyDatasetComparisonArrayRegistry

    @classmethod
    def _get_plot_registry(cls) -> Type[DummyDatasetComparisonPlotRegistry]:
        return DummyDatasetComparisonPlotRegistry

    @classmethod
    def _get_score_collection_registry(cls) -> Type[DummyDatasetComparisonScoreCollectionRegistry]:
        return DummyDatasetComparisonScoreCollectionRegistry

    @classmethod
    def _get_array_collection_registry(cls) -> Type[DummyDatasetComparisonArrayCollectionRegistry]:
        return DummyDatasetComparisonArrayCollectionRegistry

    @classmethod
    def _get_plot_collection_registry(cls) -> Type[DummyDatasetComparisonPlotCollectionRegistry]:
        return DummyDatasetComparisonPlotCollectionRegistry
