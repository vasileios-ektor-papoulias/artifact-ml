from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, Sequence, Type, TypeVar

from artifact_core.base.artifact_dependencies import ArtifactResult
from artifact_experiment.base.components.plans.base import CallbackExecutionPlan, PlanBuildContext

from artifact_torch.base.components.callbacks.backward_hook import (
    BackwardHookArrayCallback,
    BackwardHookArrayCollectionCallback,
    BackwardHookCallback,
    BackwardHookPlotCallback,
    BackwardHookPlotCollectionCallback,
    BackwardHookScoreCallback,
    BackwardHookScoreCollectionCallback,
)
from artifact_torch.base.components.callbacks.hook import HookCallbackResources
from artifact_torch.base.components.handler_suites.hook import HookCallbackHandlerSuite
from artifact_torch.base.model.base import Model


@dataclass(frozen=True)
class BackwardHookPlanBuildContext(PlanBuildContext):
    pass


ModelTContr = TypeVar("ModelTContr", bound=Model[Any, Any], contravariant=True)
BackwardHookPlanT = TypeVar("BackwardHookPlanT", bound="BackwardHookPlan[Any]")


class BackwardHookPlan(
    CallbackExecutionPlan[
        HookCallbackHandlerSuite[ModelTContr],
        BackwardHookCallback[ModelTContr, ArtifactResult, Any],
        HookCallbackResources[ModelTContr],
        BackwardHookPlanBuildContext,
    ],
    Generic[ModelTContr],
):
    @classmethod
    @abstractmethod
    def _get_score_callbacks(
        cls, context: BackwardHookPlanBuildContext
    ) -> Sequence[BackwardHookScoreCallback[ModelTContr]]: ...

    @classmethod
    @abstractmethod
    def _get_array_callbacks(
        cls, context: BackwardHookPlanBuildContext
    ) -> Sequence[BackwardHookArrayCallback[ModelTContr]]: ...

    @classmethod
    @abstractmethod
    def _get_plot_callbacks(
        cls, context: BackwardHookPlanBuildContext
    ) -> Sequence[BackwardHookPlotCallback[ModelTContr]]: ...

    @classmethod
    @abstractmethod
    def _get_score_collection_callbacks(
        cls, context: BackwardHookPlanBuildContext
    ) -> Sequence[BackwardHookScoreCollectionCallback[ModelTContr]]: ...

    @classmethod
    @abstractmethod
    def _get_array_collection_callbacks(
        cls, context: BackwardHookPlanBuildContext
    ) -> Sequence[BackwardHookArrayCollectionCallback[ModelTContr]]: ...

    @classmethod
    @abstractmethod
    def _get_plot_collection_callbacks(
        cls, context: BackwardHookPlanBuildContext
    ) -> Sequence[BackwardHookPlotCollectionCallback[ModelTContr]]: ...

    def attach(self, resources: HookCallbackResources[ModelTContr]) -> bool:
        return self._handler_suite.attach(resources=resources)

    @classmethod
    def _get_handler_suite_type(cls) -> Type[HookCallbackHandlerSuite[ModelTContr]]:
        return HookCallbackHandlerSuite
