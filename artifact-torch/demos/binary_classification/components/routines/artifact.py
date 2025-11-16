from typing import Optional, Type

import pandas as pd
from artifact_experiment.tracking import DataSplit
from artifact_torch.binary_classification import (
    BinaryClassificationPlan,
    BinaryClassificationRoutine,
)

from demos.binary_classification.components.plans.artifact import DemoBinaryClassificationPlan
from demos.binary_classification.config.constants import (
    ARTIFACT_ROUTINE_PERIOD,
    CLASSIFICATION_THRESHOLD,
)
from demos.binary_classification.contracts.workflow import WorkflowClassificationParams


class DemoBinaryClassificationRoutine(
    BinaryClassificationRoutine[WorkflowClassificationParams, pd.DataFrame]
):
    @classmethod
    def _get_period(cls, data_split: DataSplit) -> Optional[int]:
        if data_split is DataSplit.TRAIN:
            return ARTIFACT_ROUTINE_PERIOD
        elif data_split is DataSplit.VALIDATION:
            return ARTIFACT_ROUTINE_PERIOD

    @classmethod
    def _get_artifact_plan(cls, data_split: DataSplit) -> Optional[Type[BinaryClassificationPlan]]:
        if data_split is DataSplit.TRAIN:
            return DemoBinaryClassificationPlan
        elif data_split is DataSplit.VALIDATION:
            return DemoBinaryClassificationPlan

    @classmethod
    def _get_classification_params(cls) -> WorkflowClassificationParams:
        return WorkflowClassificationParams(threshold=CLASSIFICATION_THRESHOLD)
