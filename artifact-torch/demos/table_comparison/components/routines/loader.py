from typing import Any, Optional

from artifact_experiment import DataSplit
from artifact_experiment.tracking import TrackingClient
from artifact_torch.base.components.plans.forward_hook import ForwardHookPlan
from artifact_torch.base.components.plans.model_io import ModelIOPlan
from artifact_torch.base.components.routines.loader import DataLoaderRoutine
from artifact_torch.base.model.base import Model

from demos.table_comparison.components.plans.forward_hook import DemoForwardHookPlan
from demos.table_comparison.components.plans.model_io import DemoModelIOPlan
from demos.table_comparison.components.protocols import DemoModelInput, DemoModelOutput


class DemoLoaderRoutine(
    DataLoaderRoutine[Model[Any, DemoModelOutput], DemoModelInput, DemoModelOutput]
):
    @staticmethod
    def _get_model_io_plan(
        data_split: DataSplit, tracking_client: Optional[TrackingClient]
    ) -> Optional[ModelIOPlan[DemoModelInput, DemoModelOutput]]:
        if data_split is DataSplit.TRAIN:
            return DemoModelIOPlan.build(data_split=data_split, tracking_client=tracking_client)

    @staticmethod
    def _get_forward_hook_plan(
        data_split: DataSplit, tracking_client: Optional[TrackingClient]
    ) -> Optional[ForwardHookPlan[Model[Any, Any]]]:
        if data_split is DataSplit.TRAIN:
            return DemoForwardHookPlan.build(data_split=data_split, tracking_client=tracking_client)
