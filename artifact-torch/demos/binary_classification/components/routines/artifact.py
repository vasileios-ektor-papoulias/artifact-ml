from typing import List, Mapping, Optional

import pandas as pd
from artifact_core.binary_classification import (
    BinaryClassificationArrayCollectionType,
    BinaryClassificationArrayType,
    BinaryClassificationPlotCollectionType,
    BinaryClassificationPlotType,
    BinaryClassificationScoreCollectionType,
    BinaryClassificationScoreType,
    BinaryFeatureSpecProtocol,
)
from artifact_experiment import DataSplit
from artifact_experiment.binary_classification import BinaryClassificationPlan
from artifact_experiment.tracking import TrackingClient
from artifact_torch.binary_classification import BinaryClassificationRoutine

from demos.binary_classification.components.routines.protocols import DemoClassificationParams
from demos.binary_classification.config.constants import (
    ARTIFACT_VALIDATION_PERIOD,
    CLASSIFICATION_THRESHOLD,
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


class DemoBinaryClassificationRoutine(
    BinaryClassificationRoutine[DemoClassificationParams, pd.DataFrame]
):
    @classmethod
    def _get_periods(cls) -> Mapping[DataSplit, int]:
        return {DataSplit.TRAIN: ARTIFACT_VALIDATION_PERIOD}

    @classmethod
    def _get_validation_plans(
        cls,
        artifact_resource_spec: BinaryFeatureSpecProtocol,
        tracking_client: Optional[TrackingClient],
    ) -> Mapping[DataSplit, BinaryClassificationPlan]:
        return {
            DataSplit.TRAIN: DemoBinaryClassificationPlan.build(
                resource_spec=artifact_resource_spec, tracking_client=tracking_client
            )
        }

    @classmethod
    def _get_classification_params(cls) -> DemoClassificationParams:
        return DemoClassificationParams(threshold=CLASSIFICATION_THRESHOLD)
