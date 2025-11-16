from typing import Any, Optional, Type

from artifact_experiment.tracking import DataSplit
from artifact_torch.core import Model
from artifact_torch.plans import ForwardHookPlan, ModelIOPlan
from artifact_torch.routines import DataLoaderRoutine

from demos.table_comparison.components.plans.forward_hook import DemoForwardHookPlan
from demos.table_comparison.components.plans.model_io import DemoModelIOPlan
from demos.table_comparison.contracts.workflow import WorkflowInput, WorkflowOutput


class DemoLoaderRoutine(
    DataLoaderRoutine[Model[Any, WorkflowOutput], WorkflowInput, WorkflowOutput]
):
    @classmethod
    def _get_model_io_plan(
        cls, data_split: DataSplit
    ) -> Optional[Type[ModelIOPlan[WorkflowInput, WorkflowOutput]]]:
        if data_split is DataSplit.TRAIN:
            return DemoModelIOPlan

    @classmethod
    def _get_forward_hook_plan(
        cls, data_split: DataSplit
    ) -> Optional[Type[ForwardHookPlan[Model[Any, Any]]]]:
        if data_split is DataSplit.TRAIN:
            return DemoForwardHookPlan
