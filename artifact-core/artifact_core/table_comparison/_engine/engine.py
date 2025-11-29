from typing import Type

from artifact_core.table_comparison._engine.base import TableComparisonEngineBase
from artifact_core.table_comparison._registries.array_collections import (
    TableComparisonArrayCollectionRegistry,
)
from artifact_core.table_comparison._registries.arrays import TableComparisonArrayRegistry
from artifact_core.table_comparison._registries.base import (
    TableComparisonArrayCollectionRegistryBase,
    TableComparisonArrayRegistryBase,
    TableComparisonPlotCollectionRegistryBase,
    TableComparisonPlotRegistryBase,
    TableComparisonScoreCollectionRegistryBase,
    TableComparisonScoreRegistryBase,
)
from artifact_core.table_comparison._registries.plot_collections import (
    TableComparisonPlotCollectionRegistry,
)
from artifact_core.table_comparison._registries.plots import TableComparisonPlotRegistry
from artifact_core.table_comparison._registries.score_collections import (
    TableComparisonScoreCollectionRegistry,
)
from artifact_core.table_comparison._registries.scores import TableComparisonScoreRegistry
from artifact_core.table_comparison._types.array_collections import (
    TableComparisonArrayCollectionType,
)
from artifact_core.table_comparison._types.arrays import TableComparisonArrayType
from artifact_core.table_comparison._types.plot_collections import TableComparisonPlotCollectionType
from artifact_core.table_comparison._types.plots import TableComparisonPlotType
from artifact_core.table_comparison._types.score_collections import (
    TableComparisonScoreCollectionType,
)
from artifact_core.table_comparison._types.scores import TableComparisonScoreType


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
    ) -> Type[TableComparisonScoreRegistryBase[TableComparisonScoreType]]:
        return TableComparisonScoreRegistry

    @classmethod
    def _get_array_registry(
        cls,
    ) -> Type[TableComparisonArrayRegistryBase[TableComparisonArrayType]]:
        return TableComparisonArrayRegistry

    @classmethod
    def _get_plot_registry(
        cls,
    ) -> Type[TableComparisonPlotRegistryBase[TableComparisonPlotType]]:
        return TableComparisonPlotRegistry

    @classmethod
    def _get_score_collection_registry(
        cls,
    ) -> Type[TableComparisonScoreCollectionRegistryBase[TableComparisonScoreCollectionType]]:
        return TableComparisonScoreCollectionRegistry

    @classmethod
    def _get_array_collection_registry(
        cls,
    ) -> Type[TableComparisonArrayCollectionRegistryBase[TableComparisonArrayCollectionType]]:
        return TableComparisonArrayCollectionRegistry

    @classmethod
    def _get_plot_collection_registry(
        cls,
    ) -> Type[TableComparisonPlotCollectionRegistryBase[TableComparisonPlotCollectionType]]:
        return TableComparisonPlotCollectionRegistry
