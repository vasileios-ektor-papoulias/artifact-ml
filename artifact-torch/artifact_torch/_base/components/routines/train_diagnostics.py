from abc import abstractmethod
from typing import Any, Generic, Optional, Type, TypeVar

from artifact_experiment.base.tracking.background.tracking_queue import TrackingQueue

from artifact_torch.base.components.callbacks.hook import HookCallbackResources
from artifact_torch.base.components.plans.backward_hook import (
    BackwardHookPlan,
    BackwardHookPlanBuildContext,
)
from artifact_torch.base.components.plans.forward_hook import (
    ForwardHookPlan,
    ForwardHookPlanBuildContext,
)
from artifact_torch.base.components.plans.model_io import ModelIOPlan, ModelIOPlanBuildContext
from artifact_torch.base.components.routines.base import PlanExecutionRoutine, RoutineResources
from artifact_torch.base.model.base import Model
from artifact_torch.base.model.io import ModelInput, ModelOutput

ModelTContr = TypeVar("ModelTContr", bound=Model[Any, Any], contravariant=True)
ModelInputTContr = TypeVar("ModelInputTContr", bound=ModelInput, contravariant=True)
ModelOutputTContr = TypeVar("ModelOutputTContr", bound=ModelOutput, contravariant=True)
TrainDiagnosticsRoutineT = TypeVar(
    "TrainDiagnosticsRoutineT", bound="TrainDiagnosticsRoutine[Any, Any, Any]"
)


class TrainDiagnosticsRoutine(
    PlanExecutionRoutine[ModelTContr], Generic[ModelTContr, ModelInputTContr, ModelOutputTContr]
):
    _callback_trigger_identifier = "EPOCH"

    def __init__(
        self,
        model_io_plan: Optional[ModelIOPlan[ModelInputTContr, ModelOutputTContr]],
        forward_hook_plan: Optional[ForwardHookPlan[ModelTContr]],
        backward_hook_plan: Optional[BackwardHookPlan[ModelTContr]],
    ):
        self._model_io_plan = model_io_plan
        self._forward_hook_plan = forward_hook_plan
        self._backward_hook_plan = backward_hook_plan
        plans = [
            plan
            for plan in [model_io_plan, forward_hook_plan, backward_hook_plan]
            if plan is not None
        ]
        super().__init__(plans=plans)

    @classmethod
    def build(
        cls: Type[TrainDiagnosticsRoutineT], tracking_queue: Optional[TrackingQueue] = None
    ) -> TrainDiagnosticsRoutineT:
        model_io_build_context = ModelIOPlanBuildContext(tracking_queue=tracking_queue)
        forward_hook_build_context = ForwardHookPlanBuildContext(tracking_queue=tracking_queue)
        backward_hook_build_context = BackwardHookPlanBuildContext(tracking_queue=tracking_queue)
        model_io_plan = cls._build_model_io_plan(context=model_io_build_context)
        forward_hook_plan = cls._build_forward_hook_plan(context=forward_hook_build_context)
        backward_hook_plan = cls._build_backward_hook_plan(context=backward_hook_build_context)
        routine = cls(
            model_io_plan=model_io_plan,
            forward_hook_plan=forward_hook_plan,
            backward_hook_plan=backward_hook_plan,
        )
        return routine

    @classmethod
    @abstractmethod
    def _get_model_io_plan(
        cls,
    ) -> Optional[Type[ModelIOPlan[ModelInputTContr, ModelOutputTContr]]]: ...

    @classmethod
    @abstractmethod
    def _get_forward_hook_plan(cls) -> Optional[Type[ForwardHookPlan[ModelTContr]]]: ...

    @classmethod
    @abstractmethod
    def _get_backward_hook_plan(cls) -> Optional[Type[BackwardHookPlan[ModelTContr]]]: ...

    def execute(self, resources: RoutineResources[ModelTContr]):
        callback_resources = HookCallbackResources[ModelTContr](
            model=resources.model,
            step=resources.n_epochs_elapsed,
            trigger=self._callback_trigger_identifier,
        )
        if self._model_io_plan is not None:
            self._model_io_plan.execute(resources=callback_resources)
        if self._forward_hook_plan is not None:
            self._forward_hook_plan.execute(resources=callback_resources)
        if self._backward_hook_plan is not None:
            self._backward_hook_plan.execute(resources=callback_resources)

    def attach(self, model: ModelTContr, n_epochs_elapsed: int) -> bool:
        resources = HookCallbackResources[ModelTContr](model=model, step=n_epochs_elapsed)
        any_attached = False
        if self._model_io_plan is not None:
            any_attached |= self._model_io_plan.attach(resources=resources)
        if self._forward_hook_plan is not None:
            any_attached |= self._forward_hook_plan.attach(resources=resources)
        if self._backward_hook_plan is not None:
            any_attached |= self._backward_hook_plan.attach(resources=resources)
        return any_attached

    @classmethod
    def _build_model_io_plan(
        cls,
        context: ModelIOPlanBuildContext,
    ) -> Optional[ModelIOPlan[ModelInputTContr, ModelOutputTContr]]:
        plan_class = cls._get_model_io_plan()
        if plan_class is not None:
            return plan_class.build(context=context)

    @classmethod
    def _build_forward_hook_plan(
        cls,
        context: ForwardHookPlanBuildContext,
    ) -> Optional[ForwardHookPlan[ModelTContr]]:
        plan_class = cls._get_forward_hook_plan()
        if plan_class is not None:
            return plan_class.build(context=context)

    @classmethod
    def _build_backward_hook_plan(
        cls,
        context: BackwardHookPlanBuildContext,
    ) -> Optional[BackwardHookPlan[ModelTContr]]:
        plan_class = cls._get_backward_hook_plan()
        if plan_class is not None:
            return plan_class.build(context=context)
