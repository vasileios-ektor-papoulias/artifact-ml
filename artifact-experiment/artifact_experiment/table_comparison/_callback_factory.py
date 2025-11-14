from typing import Type

from artifact_core.table_comparison import (
    TableComparisonArrayCollectionType,
    TableComparisonArrayType,
    TableComparisonPlotCollectionType,
    TableComparisonPlotType,
    TableComparisonScoreCollectionType,
    TableComparisonScoreType,
)
from artifact_core.table_comparison.spi import (
    TableComparisonArrayCollectionRegistry,
    TableComparisonArrayRegistry,
    TableComparisonArtifactResources,
    TableComparisonPlotCollectionRegistry,
    TableComparisonPlotRegistry,
    TableComparisonScoreCollectionRegistry,
    TableComparisonScoreRegistry,
    TabularDataSpecProtocol,
)

from artifact_experiment._base.components.factories.artifact import ArtifactCallbackFactory


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
    def _get_score_registry() -> Type[TableComparisonScoreRegistry]:
        return TableComparisonScoreRegistry

    @staticmethod
    def _get_array_registry() -> Type[TableComparisonArrayRegistry]:
        return TableComparisonArrayRegistry

    @staticmethod
    def _get_plot_registry() -> Type[TableComparisonPlotRegistry]:
        return TableComparisonPlotRegistry

    @staticmethod
    def _get_score_collection_registry() -> Type[TableComparisonScoreCollectionRegistry]:
        return TableComparisonScoreCollectionRegistry

    @staticmethod
    def _get_array_collection_registry() -> Type[TableComparisonArrayCollectionRegistry]:
        return TableComparisonArrayCollectionRegistry

    @staticmethod
    def _get_plot_collection_registry() -> Type[TableComparisonPlotCollectionRegistry]:
        return TableComparisonPlotCollectionRegistry
