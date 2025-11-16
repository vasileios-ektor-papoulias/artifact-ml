from typing import Any, Optional, Type

from artifact_experiment.tracking import DataSplit
from artifact_torch.nn import Model
from artifact_torch.nn.plans import ForwardHookPlan, ModelIOPlan
from artifact_torch.nn.routines import DataLoaderRoutine

from demos.binary_classification.components.plans.forward_hook import DataLoaderForwardHookPlan
from demos.binary_classification.components.plans.model_io import DataLoaderModelIOPlan
from demos.binary_classification.contracts.workflow import WorkflowInput, WorkflowOutput


class DemoLoaderRoutine(DataLoaderRoutine[Model[Any, Any], WorkflowInput, WorkflowOutput]):
    @classmethod
    def _get_model_io_plan(
        cls, data_split: DataSplit
    ) -> Optional[Type[ModelIOPlan[WorkflowInput, WorkflowOutput]]]:
        if data_split is DataSplit.TRAIN:
            return DataLoaderModelIOPlan
        elif data_split is DataSplit.VALIDATION:
            return DataLoaderModelIOPlan

    @classmethod
    def _get_forward_hook_plan(
        cls, data_split: DataSplit
    ) -> Optional[Type[ForwardHookPlan[Model[Any, Any]]]]:
        if data_split is DataSplit.TRAIN:
            return DataLoaderForwardHookPlan
