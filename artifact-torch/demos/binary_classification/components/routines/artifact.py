from typing import List, Optional

from artifact_core.libs.resource_spec.binary.protocol import BinaryFeatureSpecProtocol
from artifact_experiment.base.tracking.client import TrackingClient
from artifact_experiment.binary_classification.validation_plan import (
    BinaryClassificationArrayCollectionType,
    BinaryClassificationArrayType,
    BinaryClassificationPlan,
    BinaryClassificationPlotCollectionType,
    BinaryClassificationPlotType,
    BinaryClassificationScoreCollectionType,
    BinaryClassificationScoreType,
)
from artifact_torch.binary_classification.routine import BinaryClassificationRoutine
from demos.binary_classification.config.constants import (
    ARTIFACT_VALIDATION_PERIOD,
    CLASSIFICATION_THRESHOLD,
)
from demos.binary_classification.model.classifier import MLPClassificationParams


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


class DemoBinaryClassificationRoutine(BinaryClassificationRoutine[MLPClassificationParams]):
    @classmethod
    def _get_period(cls) -> int:
        return ARTIFACT_VALIDATION_PERIOD

    @classmethod
    def _get_generation_params(cls) -> MLPClassificationParams:
        return MLPClassificationParams(threshold=CLASSIFICATION_THRESHOLD)

    @classmethod
    def _get_validation_plan(
        cls,
        artifact_resource_spec: BinaryFeatureSpecProtocol,
        tracking_client: Optional[TrackingClient],
    ) -> BinaryClassificationPlan:
        return DemoBinaryClassificationPlan.build(
            resource_spec=artifact_resource_spec, tracking_client=tracking_client
        )
