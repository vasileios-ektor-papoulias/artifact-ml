from typing import List

from artifact_torch.binary_classification import (
    BinaryClassificationArrayCollectionType,
    BinaryClassificationArrayType,
    BinaryClassificationPlan,
    BinaryClassificationPlotCollectionType,
    BinaryClassificationPlotType,
    BinaryClassificationScoreCollectionType,
    BinaryClassificationScoreType,
)


class DemoBinaryClassificationPlan(BinaryClassificationPlan):
    @staticmethod
    def _get_score_types() -> List[BinaryClassificationScoreType]:
        return [
            BinaryClassificationScoreType.PRECISION,
            BinaryClassificationScoreType.RECALL,
            BinaryClassificationScoreType.ROC_AUC,
            BinaryClassificationScoreType.PR_AUC,
        ]

    @staticmethod
    def _get_array_types() -> List[BinaryClassificationArrayType]:
        return []

    @staticmethod
    def _get_plot_types() -> List[BinaryClassificationPlotType]:
        return [
            BinaryClassificationPlotType.SCORE_PDF,
            BinaryClassificationPlotType.GROUND_TRUTH_PROB_PDF,
        ]

    @staticmethod
    def _get_score_collection_types() -> List[BinaryClassificationScoreCollectionType]:
        return [
            BinaryClassificationScoreCollectionType.SCORE_MEANS,
            BinaryClassificationScoreCollectionType.SCORE_STDS,
            BinaryClassificationScoreCollectionType.SCORE_STATS,
        ]

    @staticmethod
    def _get_array_collection_types() -> List[BinaryClassificationArrayCollectionType]:
        return []

    @staticmethod
    def _get_plot_collection_types() -> List[BinaryClassificationPlotCollectionType]:
        return [
            BinaryClassificationPlotCollectionType.THRESHOLD_VARIATION_CURVES,
            BinaryClassificationPlotCollectionType.CONFUSION_MATRIX_PLOTS,
        ]
