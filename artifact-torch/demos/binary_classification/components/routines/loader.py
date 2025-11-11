from typing import Any, Optional, Type

from artifact_experiment import DataSplit
from artifact_torch.base.components.plans.forward_hook import ForwardHookPlan
from artifact_torch.base.components.plans.model_io import ModelIOPlan
from artifact_torch.base.components.routines.loader import DataLoaderRoutine
from artifact_torch.base.model.base import Model

from demos.binary_classification.components.plans.forward_hook import DataLoaderForwardHookPlan
from demos.binary_classification.components.plans.model_io import DataLoaderModelIOPlan
from demos.binary_classification.components.protocols import DemoModelInput, DemoModelOutput


class DemoLoaderRoutine(DataLoaderRoutine[Model[Any, Any], DemoModelInput, DemoModelOutput]):
    @classmethod
    def _get_model_io_plan(
        cls, data_split: DataSplit
    ) -> Optional[Type[ModelIOPlan[DemoModelInput, DemoModelOutput]]]:
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
