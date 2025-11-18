from abc import abstractmethod
from typing import Any, Generic, List, Mapping, Optional, Type, TypeVar

import torch
from artifact_experiment.tracking import DataSplit
from artifact_experiment.tracking.spi import TrackingQueue
from tqdm import tqdm

from artifact_torch._base.components.plans.forward_hook import (
    ForwardHookPlan,
    ForwardHookPlanBuildContext,
)
from artifact_torch._base.components.plans.model_io import ModelIOPlan, ModelIOPlanBuildContext
from artifact_torch._base.components.resources.hook import HookCallbackResources
from artifact_torch._base.components.routines.routine import PlanExecutionRoutine, RoutineResources
from artifact_torch._base.data.data_loader import DataLoader
from artifact_torch._base.model.base import Model
from artifact_torch._base.model.io import ModelInput, ModelOutput
from artifact_torch._utils.collections.key_selector import KeySelector

ModelTContr = TypeVar("ModelTContr", bound=Model[Any, Any], contravariant=True)
ModelInputTContr = TypeVar("ModelInputTContr", bound=ModelInput, contravariant=True)
ModelOutputTContr = TypeVar("ModelOutputTContr", bound=ModelOutput, contravariant=True)
DataLoaderRoutineT = TypeVar("DataLoaderRoutineT", bound="DataLoaderRoutine[Any, Any, Any]")


class DataLoaderRoutine(
    PlanExecutionRoutine[ModelTContr], Generic[ModelTContr, ModelInputTContr, ModelOutputTContr]
):
    _verbose = True
    _progressbar_message = "Processing Data Loader"

    def __init__(
        self,
        model_io_plans: Mapping[DataSplit, ModelIOPlan[ModelInputTContr, ModelOutputTContr]],
        forward_hook_plans: Mapping[DataSplit, ForwardHookPlan[ModelTContr]],
        data_loaders: Mapping[DataSplit, DataLoader[ModelInputTContr]],
    ):
        self._data_loaders = data_loaders
        self._model_io_plans = KeySelector.restrict_to_keys(model_io_plans, keys_from=data_loaders)
        self._forward_hook_plans = KeySelector.restrict_to_keys(
            forward_hook_plans, keys_from=data_loaders
        )
        plans = list(self._model_io_plans.values()) + list(self._forward_hook_plans.values())
        super().__init__(plans=plans)

    @classmethod
    def build(
        cls: Type[DataLoaderRoutineT],
        data_loaders: Mapping[DataSplit, DataLoader[ModelInputTContr]],
        tracking_queue: Optional[TrackingQueue] = None,
    ) -> DataLoaderRoutineT:
        model_io_build_context = ModelIOPlanBuildContext(tracking_queue=tracking_queue)
        forward_hook_build_context = ForwardHookPlanBuildContext(tracking_queue=tracking_queue)
        model_io_plans = {
            data_split: plan
            for data_split in DataSplit
            if (
                plan := cls._build_model_io_plan(
                    data_split=data_split, context=model_io_build_context
                )
            )
            is not None
        }
        forward_hook_plans = {
            data_split: plan
            for data_split in DataSplit
            if (
                plan := cls._build_forward_hook_plan(
                    data_split=data_split, context=forward_hook_build_context
                )
            )
            is not None
        }
        routine = cls(
            model_io_plans=model_io_plans,
            forward_hook_plans=forward_hook_plans,
            data_loaders=data_loaders,
        )
        return routine

    @property
    def _data_splits(self) -> List[DataSplit]:
        return list(self._data_loaders.keys())

    @classmethod
    @abstractmethod
    def _get_model_io_plan(
        cls, data_split: DataSplit
    ) -> Optional[Type[ModelIOPlan[ModelInputTContr, ModelOutputTContr]]]: ...

    @classmethod
    @abstractmethod
    def _get_forward_hook_plan(
        cls, data_split: DataSplit
    ) -> Optional[Type[ForwardHookPlan[ModelTContr]]]: ...

    def execute(self, resources: RoutineResources[ModelTContr]):
        for data_split in self._data_splits:
            callback_resources = HookCallbackResources[ModelTContr](
                model=resources.model, step=resources.n_epochs_elapsed, data_split=data_split
            )
            data_loader = self._data_loaders[data_split]
            model_io_plan = self._model_io_plans.get(data_split)
            forward_hook_plan = self._forward_hook_plans.get(data_split)
            self._execute(
                callback_resources=callback_resources,
                model_io_plan=model_io_plan,
                forward_hook_plan=forward_hook_plan,
                data_loader=data_loader,
            )

    @classmethod
    def _execute(
        cls,
        callback_resources: HookCallbackResources[ModelTContr],
        model_io_plan: Optional[ModelIOPlan[ModelInputTContr, ModelOutputTContr]],
        forward_hook_plan: Optional[ForwardHookPlan[ModelTContr]],
        data_loader: DataLoader[ModelInputTContr],
    ):
        any_attached = cls._attach(
            callback_resources=callback_resources,
            model_io_plan=model_io_plan,
            forward_hook_plan=forward_hook_plan,
        )
        if any_attached:
            cls._process_data_loader(model=callback_resources.model, data_loader=data_loader)
            if model_io_plan is not None:
                model_io_plan.execute(resources=callback_resources)
            if forward_hook_plan is not None:
                forward_hook_plan.execute(resources=callback_resources)

    @staticmethod
    def _attach(
        callback_resources: HookCallbackResources[ModelTContr],
        model_io_plan: Optional[ModelIOPlan[ModelInputTContr, ModelOutputTContr]],
        forward_hook_plan: Optional[ForwardHookPlan[ModelTContr]],
    ) -> bool:
        any_attached = False
        if model_io_plan is not None:
            any_attached |= model_io_plan.attach(resources=callback_resources)
        if forward_hook_plan is not None:
            any_attached |= forward_hook_plan.attach(resources=callback_resources)
        return any_attached

    @classmethod
    def _process_data_loader(cls, model: ModelTContr, data_loader: DataLoader[ModelInputTContr]):
        model.eval()
        data_loader.device = model.device
        with torch.no_grad():
            for model_input in tqdm(
                data_loader,
                desc=cls._progressbar_message,
                disable=not cls._verbose,
                leave=False,
            ):
                _ = model(model_input)

    @classmethod
    def _build_model_io_plan(
        cls, data_split: DataSplit, context: ModelIOPlanBuildContext
    ) -> Optional[ModelIOPlan[ModelInputTContr, ModelOutputTContr]]:
        plan_class = cls._get_model_io_plan(data_split=data_split)
        if plan_class is not None:
            return plan_class.build(context=context)

    @classmethod
    def _build_forward_hook_plan(
        cls, data_split: DataSplit, context: ForwardHookPlanBuildContext
    ) -> Optional[ForwardHookPlan[ModelTContr]]:
        plan_class = cls._get_forward_hook_plan(data_split=data_split)
        if plan_class is not None:
            return plan_class.build(context=context)
