from typing import Any, Optional, Type

from artifact_torch.base.components.plans.backward_hook import BackwardHookPlan
from artifact_torch.base.components.plans.forward_hook import ForwardHookPlan
from artifact_torch.base.components.plans.model_io import ModelIOPlan
from artifact_torch.base.components.routines.train_diagnostics import TrainDiagnosticsRoutine
from artifact_torch.base.model.base import Model

from demos.binary_classification.components.plans.model_io import DataLoaderModelIOPlan
from demos.binary_classification.components.protocols import DemoModelInput, DemoModelOutput


class DemoTrainDiagnosticsRoutine(
    TrainDiagnosticsRoutine[Model[Any, DemoModelOutput], DemoModelInput, DemoModelOutput]
):
    @classmethod
    def _get_model_io_plan(cls) -> Optional[Type[ModelIOPlan[DemoModelInput, DemoModelOutput]]]:
        return DataLoaderModelIOPlan

    @classmethod
    def _get_forward_hook_plan(cls) -> Optional[Type[ForwardHookPlan[Model[Any, Any]]]]:
        pass

    @classmethod
    def _get_backward_hook_plan(cls) -> Optional[Type[BackwardHookPlan[Model[Any, Any]]]]:
        pass
