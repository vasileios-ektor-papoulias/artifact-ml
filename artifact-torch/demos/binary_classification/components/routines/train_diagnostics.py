from typing import Any, Optional

from artifact_experiment.tracking import TrackingClient
from artifact_torch.base.components.plans.backward_hook import BackwardHookPlan
from artifact_torch.base.components.plans.forward_hook import ForwardHookPlan
from artifact_torch.base.components.plans.model_io import ModelIOPlan
from artifact_torch.base.components.routines.train_diagnostics import TrainDiagnosticsRoutine
from artifact_torch.base.model.base import Model

from demos.binary_classification.components.plans.forward_hook import (
    TrainDiagnosticsForwardHookPlan,
)
from demos.binary_classification.components.plans.model_io import (
    TrainDiagnosticsModelIOPlan,
)
from demos.binary_classification.components.protocols import DemoModelInput, DemoModelOutput


class DemoTrainDiagnosticsRoutine(
    TrainDiagnosticsRoutine[Model[Any, Any], DemoModelInput, DemoModelOutput]
):
    @staticmethod
    def _get_model_io_plan(
        tracking_client: Optional[TrackingClient],
    ) -> Optional[ModelIOPlan[DemoModelInput, DemoModelOutput]]:
        return TrainDiagnosticsModelIOPlan.build(tracking_client=tracking_client)

    @staticmethod
    def _get_forward_hook_plan(
        tracking_client: Optional[TrackingClient],
    ) -> Optional[ForwardHookPlan[Model[Any, Any]]]:
        return TrainDiagnosticsForwardHookPlan.build(tracking_client=tracking_client)

    @staticmethod
    def _get_backward_hook_plan(
        tracking_client: Optional[TrackingClient],
    ) -> Optional[BackwardHookPlan[Model[Any, Any]]]:
        _ = tracking_client
