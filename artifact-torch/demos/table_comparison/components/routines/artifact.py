from typing import Optional, Type

from artifact_experiment.table_comparison import TableComparisonPlan
from artifact_experiment.tracking import DataSplit
from artifact_torch.table_comparison import TableComparisonRoutine

from demos.table_comparison.components.plans.artifact import DemoTableComparisonPlan
from demos.table_comparison.config.constants import (
    ARTIFACT_ROUTINE_PERIOD,
    GENERATION_N_RECORDS,
    GENERATION_TEMPERATURE,
)
from demos.table_comparison.contracts.workflow import WorkflowGenerationParams


class DemoTableComparisonRoutine(TableComparisonRoutine[WorkflowGenerationParams]):
    @classmethod
    def _get_period(cls, data_split: DataSplit) -> Optional[int]:
        if data_split is DataSplit.TRAIN:
            return ARTIFACT_ROUTINE_PERIOD

    @classmethod
    def _get_generation_params(cls) -> WorkflowGenerationParams:
        return WorkflowGenerationParams(
            n_records=GENERATION_N_RECORDS, temperature=GENERATION_TEMPERATURE
        )

    @classmethod
    def _get_artifact_plan(cls, data_split: DataSplit) -> Optional[Type[TableComparisonPlan]]:
        if data_split is DataSplit.TRAIN:
            return DemoTableComparisonPlan
