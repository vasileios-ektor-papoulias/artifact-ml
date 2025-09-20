from typing import Type

from artifact_core.table_comparison.engine.base import (
    TableComparisonEngineBase,
)
from artifact_core.table_comparison.registries.array_collections.registry import (
    TableComparisonArrayCollectionRegistry,
    TableComparisonArrayCollectionType,
)
from artifact_core.table_comparison.registries.arrays.registry import (
    TableComparisonArrayRegistry,
    TableComparisonArrayType,
)
from artifact_core.table_comparison.registries.base import (
    TableComparisonArrayCollectionRegistryBase,
    TableComparisonArrayRegistryBase,
    TableComparisonPlotCollectionRegistryBase,
    TableComparisonPlotRegistryBase,
    TableComparisonScoreRegistryBase,
)
from artifact_core.table_comparison.registries.plot_collections.registry import (
    TableComparisonPlotCollectionRegistry,
    TableComparisonPlotCollectionType,
)
from artifact_core.table_comparison.registries.plots.registry import (
    TableComparisonPlotRegistry,
    TableComparisonPlotType,
)
from artifact_core.table_comparison.registries.score_collections.registry import (
    TableComparisonScoreCollectionRegistry,
    TableComparisonScoreCollectionType,
)
from artifact_core.table_comparison.registries.scores.registry import (
    TableComparisonScoreRegistry,
    TableComparisonScoreType,
)


class TableComparisonEngine(
    TableComparisonEngineBase[
        TableComparisonScoreType,
        TableComparisonArrayType,
        TableComparisonPlotType,
        TableComparisonScoreCollectionType,
        TableComparisonArrayCollectionType,
        TableComparisonPlotCollectionType,
    ]
):
    @classmethod
    def _get_score_registry(
        cls,
    ) -> Type[TableComparisonScoreRegistryBase]:
        return TableComparisonScoreRegistry

    @classmethod
    def _get_array_registry(
        cls,
    ) -> Type[TableComparisonArrayRegistryBase]:
        return TableComparisonArrayRegistry

    @classmethod
    def _get_plot_registry(
        cls,
    ) -> Type[TableComparisonPlotRegistryBase]:
        return TableComparisonPlotRegistry

    @classmethod
    def _get_score_collection_registry(
        cls,
    ) -> Type[TableComparisonScoreCollectionRegistry]:
        return TableComparisonScoreCollectionRegistry

    @classmethod
    def _get_array_collection_registry(
        cls,
    ) -> Type[TableComparisonArrayCollectionRegistryBase]:
        return TableComparisonArrayCollectionRegistry

    @classmethod
    def _get_plot_collection_registry(
        cls,
    ) -> Type[TableComparisonPlotCollectionRegistryBase]:
        return TableComparisonPlotCollectionRegistry
