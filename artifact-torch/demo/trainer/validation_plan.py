from typing import List

from artifact_experiment.table_comparison.validation_plan import (
    TableComparisonArrayCollectionType,
    TableComparisonArrayType,
    TableComparisonPlan,
    TableComparisonPlotCollectionType,
    TableComparisonPlotType,
    TableComparisonScoreCollectionType,
    TableComparisonScoreType,
)


class TabularVAEValidationPlan(TableComparisonPlan):
    @staticmethod
    def _get_score_types() -> List[TableComparisonScoreType]:
        return [
            TableComparisonScoreType.MEAN_JS_DISTANCE,
            TableComparisonScoreType.PAIRWISE_CORRELATION_DISTANCE,
        ]

    @staticmethod
    def _get_array_types() -> List[TableComparisonArrayType]:
        return []

    @staticmethod
    def _get_plot_types() -> List[TableComparisonPlotType]:
        return [
            TableComparisonPlotType.PDF_PLOT,
            TableComparisonPlotType.CDF_PLOT,
            TableComparisonPlotType.DESCRIPTIVE_STATS_COMPARISON_PLOT,
            TableComparisonPlotType.PCA_PROJECTION_PLOT,
            TableComparisonPlotType.PAIRWISE_CORRELATION_COMPARISON_HEATMAP,
        ]

    @staticmethod
    def _get_score_collection_types() -> List[TableComparisonScoreCollectionType]:
        return [
            TableComparisonScoreCollectionType.JS_DISTANCE,
        ]

    @staticmethod
    def _get_array_collection_types() -> List[TableComparisonArrayCollectionType]:
        return [
            TableComparisonArrayCollectionType.MEANS,
            TableComparisonArrayCollectionType.STDS,
            TableComparisonArrayCollectionType.MINIMA,
            TableComparisonArrayCollectionType.MAXIMA,
        ]

    @staticmethod
    def _get_plot_collection_types() -> List[TableComparisonPlotCollectionType]:
        return [
            TableComparisonPlotCollectionType.PDF_PLOTS,
            TableComparisonPlotCollectionType.CDF_PLOTS,
        ]
