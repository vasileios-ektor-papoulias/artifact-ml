from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, Sequence, Type, TypeVar

from artifact_core._base.artifact_dependencies import ArtifactResult
from artifact_experiment.base.components.plans.base import CallbackExecutionPlan, PlanBuildContext

from artifact_torch.base.components.callbacks.forward_hook import (
    ForwardHookArrayCallback,
    ForwardHookArrayCollectionCallback,
    ForwardHookCallback,
    ForwardHookPlotCallback,
    ForwardHookPlotCollectionCallback,
    ForwardHookScoreCallback,
    ForwardHookScoreCollectionCallback,
)
from artifact_torch.base.components.callbacks.hook import HookCallbackResources
from artifact_torch.base.components.handler_suites.hook import HookCallbackHandlerSuite
from artifact_torch.base.model.base import Model


@dataclass(frozen=True)
class ForwardHookPlanBuildContext(PlanBuildContext):
    pass


ModelTContr = TypeVar("ModelTContr", bound=Model[Any, Any], contravariant=True)
ForwardHookPlanT = TypeVar("ForwardHookPlanT", bound="ForwardHookPlan[Any]")


class ForwardHookPlan(
    CallbackExecutionPlan[
        HookCallbackHandlerSuite[ModelTContr],
        ForwardHookCallback[ModelTContr, ArtifactResult, Any],
        HookCallbackResources[ModelTContr],
        ForwardHookPlanBuildContext,
    ],
    Generic[ModelTContr],
):
    @classmethod
    @abstractmethod
    def _get_score_callbacks(
        cls, context: ForwardHookPlanBuildContext
    ) -> Sequence[ForwardHookScoreCallback[ModelTContr]]: ...

    @classmethod
    @abstractmethod
    def _get_array_callbacks(
        cls, context: ForwardHookPlanBuildContext
    ) -> Sequence[ForwardHookArrayCallback[ModelTContr]]: ...

    @classmethod
    @abstractmethod
    def _get_plot_callbacks(
        cls, context: ForwardHookPlanBuildContext
    ) -> Sequence[ForwardHookPlotCallback[ModelTContr]]: ...

    @classmethod
    @abstractmethod
    def _get_score_collection_callbacks(
        cls, context: ForwardHookPlanBuildContext
    ) -> Sequence[ForwardHookScoreCollectionCallback[ModelTContr]]: ...

    @classmethod
    @abstractmethod
    def _get_array_collection_callbacks(
        cls, context: ForwardHookPlanBuildContext
    ) -> Sequence[ForwardHookArrayCollectionCallback[ModelTContr]]: ...

    @classmethod
    @abstractmethod
    def _get_plot_collection_callbacks(
        cls, context: ForwardHookPlanBuildContext
    ) -> Sequence[ForwardHookPlotCollectionCallback[ModelTContr]]: ...

    def attach(self, resources: HookCallbackResources[ModelTContr]) -> bool:
        return self._handler_suite.attach(resources=resources)

    @classmethod
    def _get_handler_suite_type(cls) -> Type[HookCallbackHandlerSuite[ModelTContr]]:
        return HookCallbackHandlerSuite
