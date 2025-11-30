from typing import Type

from artifact_core._domains.dataset_comparison.engine import DatasetComparisonEngine

from tests._domains.dataset_comparison.dummy.registries import (
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
from tests._domains.dataset_comparison.dummy.resource_spec import DummyResourceSpec
from tests._domains.dataset_comparison.dummy.resources import DummyDataset


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
