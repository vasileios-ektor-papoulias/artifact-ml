from typing import Optional

from artifact_core.table_comparison import (
    TabularDataSpecProtocol,
)
from artifact_experiment import DataSplit
from artifact_experiment.table_comparison import TableComparisonPlan
from artifact_experiment.tracking import TrackingClient
from artifact_torch.table_comparison import TableComparisonRoutine

from demos.table_comparison.components.plans.artifact import DemoTableComparisonPlan
from demos.table_comparison.components.protocols import (
    DemoGenerationParams,
)
from demos.table_comparison.config.constants import (
    ARTIFACT_ROUTINE_PERIOD,
    GENERATION_N_RECORDS,
    GENERATION_TEMPERATURE,
)


class DemoTableComparisonRoutine(TableComparisonRoutine[DemoGenerationParams]):
    @classmethod
    def _get_period(cls, data_split: DataSplit) -> Optional[int]:
        if data_split is DataSplit.TRAIN:
            return ARTIFACT_ROUTINE_PERIOD

    @classmethod
    def _get_artifact_plan(
        cls,
        artifact_resource_spec: TabularDataSpecProtocol,
        data_split: DataSplit,
        tracking_client: Optional[TrackingClient],
    ) -> Optional[TableComparisonPlan]:
        if data_split is DataSplit.TRAIN:
            return DemoTableComparisonPlan.build(
                resource_spec=artifact_resource_spec,
                data_split=DataSplit.TRAIN,
                tracking_client=tracking_client,
            )

    @classmethod
    def _get_generation_params(cls) -> DemoGenerationParams:
        return DemoGenerationParams(
            n_records=GENERATION_N_RECORDS, temperature=GENERATION_TEMPERATURE
        )
