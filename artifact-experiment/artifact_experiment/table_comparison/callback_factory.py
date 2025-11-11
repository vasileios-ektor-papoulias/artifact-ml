from typing import Type

from artifact_core.table_comparison._artifacts.base import (
    TableComparisonArtifactResources,
    TabularDataSpecProtocol,
)
from artifact_core.table_comparison._registries.array_collections.registry import (
    TableComparisonArrayCollectionRegistry,
    TableComparisonArrayCollectionRegistryBase,
    TableComparisonArrayCollectionType,
)
from artifact_core.table_comparison._registries.arrays.registry import (
    TableComparisonArrayRegistry,
    TableComparisonArrayRegistryBase,
    TableComparisonArrayType,
)
from artifact_core.table_comparison._registries.plot_collections.registry import (
    TableComparisonPlotCollectionRegistry,
    TableComparisonPlotCollectionRegistryBase,
    TableComparisonPlotCollectionType,
)
from artifact_core.table_comparison._registries.plots.registry import (
    TableComparisonPlotRegistry,
    TableComparisonPlotRegistryBase,
    TableComparisonPlotType,
)
from artifact_core.table_comparison._registries.score_collections.registry import (
    TableComparisonScoreCollectionRegistry,
    TableComparisonScoreCollectionRegistryBase,
    TableComparisonScoreCollectionType,
)
from artifact_core.table_comparison._registries.scores.registry import (
    TableComparisonScoreRegistry,
    TableComparisonScoreRegistryBase,
    TableComparisonScoreType,
)

from artifact_experiment.base.components.factories.artifact import ArtifactCallbackFactory


class TableComparisonCallbackFactory(
    ArtifactCallbackFactory[
        TableComparisonArtifactResources,
        TabularDataSpecProtocol,
        TableComparisonScoreType,
        TableComparisonArrayType,
        TableComparisonPlotType,
        TableComparisonScoreCollectionType,
        TableComparisonArrayCollectionType,
        TableComparisonPlotCollectionType,
    ]
):
    @staticmethod
    def _get_score_registry() -> Type[TableComparisonScoreRegistryBase[TableComparisonScoreType]]:
        return TableComparisonScoreRegistry

    @staticmethod
    def _get_array_registry() -> Type[TableComparisonArrayRegistryBase[TableComparisonArrayType]]:
        return TableComparisonArrayRegistry

    @staticmethod
    def _get_plot_registry() -> Type[TableComparisonPlotRegistryBase[TableComparisonPlotType]]:
        return TableComparisonPlotRegistry

    @staticmethod
    def _get_score_collection_registry() -> Type[
        TableComparisonScoreCollectionRegistryBase[TableComparisonScoreCollectionType]
    ]:
        return TableComparisonScoreCollectionRegistry

    @staticmethod
    def _get_array_collection_registry() -> Type[
        TableComparisonArrayCollectionRegistryBase[TableComparisonArrayCollectionType]
    ]:
        return TableComparisonArrayCollectionRegistry

    @staticmethod
    def _get_plot_collection_registry() -> Type[
        TableComparisonPlotCollectionRegistryBase[TableComparisonPlotCollectionType]
    ]:
        return TableComparisonPlotCollectionRegistry
