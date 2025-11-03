from typing import List

from artifact_core.table_comparison import (
    TableComparisonArrayCollectionType,
    TableComparisonArrayType,
    TableComparisonPlotCollectionType,
    TableComparisonPlotType,
    TableComparisonScoreCollectionType,
    TableComparisonScoreType,
)
from artifact_experiment.table_comparison import TableComparisonPlan


class DemoTableComparisonPlan(TableComparisonPlan):
    @staticmethod
    def _get_score_types() -> List[TableComparisonScoreType]:
        return [
            TableComparisonScoreType.MEAN_JS_DISTANCE,
            TableComparisonScoreType.CORRELATION_DISTANCE,
        ]

    @staticmethod
    def _get_array_types() -> List[TableComparisonArrayType]:
        return []

    @staticmethod
    def _get_plot_types() -> List[TableComparisonPlotType]:
        return [
            TableComparisonPlotType.PDF,
            TableComparisonPlotType.CDF,
            TableComparisonPlotType.DESCRIPTIVE_STATS_ALIGNMENT,
            TableComparisonPlotType.PCA_JUXTAPOSITION,
            TableComparisonPlotType.CORRELATION_HEATMAP_JUXTAPOSITION,
        ]

    @staticmethod
    def _get_score_collection_types() -> List[TableComparisonScoreCollectionType]:
        return [
            TableComparisonScoreCollectionType.JS_DISTANCE,
        ]

    @staticmethod
    def _get_array_collection_types() -> List[TableComparisonArrayCollectionType]:
        return [
            TableComparisonArrayCollectionType.MEAN_JUXTAPOSITION,
            TableComparisonArrayCollectionType.STD_JUXTAPOSITION,
            TableComparisonArrayCollectionType.MIN_JUXTAPOSITION,
            TableComparisonArrayCollectionType.MAX_JUXTAPOSITION,
        ]

    @staticmethod
    def _get_plot_collection_types() -> List[TableComparisonPlotCollectionType]:
        return [
            TableComparisonPlotCollectionType.PDF,
            TableComparisonPlotCollectionType.CDF,
        ]
