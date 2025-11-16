from typing import Any, Optional, Type

from artifact_torch.core import Model
from artifact_torch.plans import BackwardHookPlan, ForwardHookPlan, ModelIOPlan
from artifact_torch.routines import TrainDiagnosticsRoutine

from demos.binary_classification.components.plans.model_io import TrainDiagnosticsModelIOPlan
from demos.binary_classification.contracts.workflow import WorkflowInput, WorkflowOutput


class DemoTrainDiagnosticsRoutine(
    TrainDiagnosticsRoutine[Model[Any, Any], WorkflowInput, WorkflowOutput]
):
    @classmethod
    def _get_model_io_plan(cls) -> Optional[Type[ModelIOPlan[WorkflowInput, WorkflowOutput]]]:
        return TrainDiagnosticsModelIOPlan

    @classmethod
    def _get_forward_hook_plan(cls) -> Optional[Type[ForwardHookPlan[Model[Any, Any]]]]:
        pass

    @classmethod
    def _get_backward_hook_plan(cls) -> Optional[Type[BackwardHookPlan[Model[Any, Any]]]]:
        pass
