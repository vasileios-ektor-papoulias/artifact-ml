from typing import List, Optional

from artifact_core.libs.resource_spec.tabular.protocol import TabularDataSpecProtocol
from artifact_experiment.base.tracking.client import TrackingClient
from artifact_torch.base.components.callbacks.data_loader import (
    DataLoaderScoreCallback,
)
from artifact_torch.libs.components.callbacks.data_loader.loss import (
    TrainLossCallback,
)
from artifact_torch.table_comparison.validation_routine import TableComparisonValidationRoutine

from demo.config.constants import (
    GENERATION_NUM_SAMPLES,
    GENERATION_USE_MEAN,
    TRAIN_LOADER_CALLBACK_PERIOD,
    VALIDATION_PLAN_CALLBACK_PERIOD,
)
from demo.model.io import TabularVAEInput, TabularVAEOutput
from demo.model.synthesizer import TabularVAEGenerationParams
from demo.trainer.validation_plan import TabularVAEValidationPlan


class TabularVAEValidationRoutine(
    TableComparisonValidationRoutine[
        TabularVAEInput,
        TabularVAEOutput,
        TabularVAEGenerationParams,
    ]
):
    @staticmethod
    def _get_generation_params() -> TabularVAEGenerationParams:
        return TabularVAEGenerationParams(
            num_samples=GENERATION_NUM_SAMPLES, use_mean=GENERATION_USE_MEAN
        )

    @staticmethod
    def _get_validation_plan(
        tabular_data_spec: TabularDataSpecProtocol,
        tracking_client: Optional[TrackingClient],
    ) -> TabularVAEValidationPlan:
        return TabularVAEValidationPlan.build(
            resource_spec=tabular_data_spec, tracking_client=tracking_client
        )

    @staticmethod
    def _get_artifact_validation_period() -> int:
        return VALIDATION_PLAN_CALLBACK_PERIOD

    @staticmethod
    def _get_train_loader_score_callbacks() -> List[
        DataLoaderScoreCallback[TabularVAEInput, TabularVAEOutput]
    ]:
        return [TrainLossCallback(period=TRAIN_LOADER_CALLBACK_PERIOD)]

    @staticmethod
    def _get_val_loader_score_callbacks() -> List[
        DataLoaderScoreCallback[TabularVAEInput, TabularVAEOutput]
    ]:
        return []
