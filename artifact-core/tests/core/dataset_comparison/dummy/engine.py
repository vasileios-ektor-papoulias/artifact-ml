from artifact_core._core.dataset_comparison.engine import DatasetComparisonEngine

from tests.core.dataset_comparison.dummy.artifact_dependencies import (
    DummyDataset,
    DummyResourceSpec,
)
from tests.core.dataset_comparison.dummy.registries import (
    DummyDatasetComparisonArrayCollectionRegistry,
    DummyDatasetComparisonArrayCollectionType,
    DummyDatasetComparisonArrayRegistry,
    DummyDatasetComparisonArrayType,
    DummyDatasetComparisonPlotCollectionRegistry,
    DummyDatasetComparisonPlotCollectionType,
    DummyDatasetComparisonPlotRegistry,
    DummyDatasetComparisonPlotType,
    DummyDatasetComparisonScoreCollectionRegistry,
    DummyDatasetComparisonScoreCollectionType,
    DummyDatasetComparisonScoreRegistry,
    DummyDatasetComparisonScoreType,
)


class DummyDatasetComparisonEngine(
    DatasetComparisonEngine[
        DummyDataset,
        DummyResourceSpec,
        DummyDatasetComparisonScoreType,
        DummyDatasetComparisonArrayType,
        DummyDatasetComparisonPlotType,
        DummyDatasetComparisonScoreCollectionType,
        DummyDatasetComparisonArrayCollectionType,
        DummyDatasetComparisonPlotCollectionType,
    ]
):
    @classmethod
    def _get_score_registry(cls):
        return DummyDatasetComparisonScoreRegistry

    @classmethod
    def _get_array_registry(cls):
        return DummyDatasetComparisonArrayRegistry

    @classmethod
    def _get_plot_registry(cls):
        return DummyDatasetComparisonPlotRegistry

    @classmethod
    def _get_score_collection_registry(cls):
        return DummyDatasetComparisonScoreCollectionRegistry

    @classmethod
    def _get_array_collection_registry(cls):
        return DummyDatasetComparisonArrayCollectionRegistry

    @classmethod
    def _get_plot_collection_registry(cls):
        return DummyDatasetComparisonPlotCollectionRegistry
