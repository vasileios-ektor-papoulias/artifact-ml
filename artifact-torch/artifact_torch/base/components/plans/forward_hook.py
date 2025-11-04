from abc import abstractmethod
from typing import Any, Generic, Optional, Sequence, Type, TypeVar

from artifact_experiment.base.entities.data_split import DataSplit
from artifact_experiment.base.plans.base import CallbackExecutionPlan
from artifact_experiment.base.tracking.client import TrackingClient

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
from artifact_torch.base.components.handlers.hook import (
    HookCallbackHandler,
    HookCallbackHandlerSuite,
)
from artifact_torch.base.model.base import Model

ModelTContr = TypeVar("ModelTContr", bound=Model[Any, Any], contravariant=True)
ForwardHookPlanT = TypeVar("ForwardHookPlanT", bound="ForwardHookPlan[Any]")


class ForwardHookPlan(
    CallbackExecutionPlan[
        HookCallbackHandler[ModelTContr, Any],
        ForwardHookCallback[ModelTContr, Any, Any],
        HookCallbackResources[ModelTContr],
    ],
    Generic[ModelTContr],
):
    def __init__(
        self,
        handler_suite: HookCallbackHandlerSuite[ModelTContr],
        data_split: Optional[DataSplit] = None,
        tracking_client: Optional[TrackingClient] = None,
    ):
        super().__init__(
            handler_suite=handler_suite,
            data_split=data_split,
            tracking_client=tracking_client,
        )

    @classmethod
    def build(
        cls: Type[ForwardHookPlanT], tracking_client: Optional[TrackingClient] = None
    ) -> ForwardHookPlanT:
        score_callbacks = cls._get_score_callbacks()
        array_callbacks = cls._get_array_callbacks()
        plot_callbacks = cls._get_plot_callbacks()
        score_collection_callbacks = cls._get_score_collection_callbacks()
        array_collection_callbacks = cls._get_array_collection_callbacks()
        plot_collection_callbacks = cls._get_plot_collection_callbacks()
        handler_suite = HookCallbackHandlerSuite[ModelTContr].build(
            score_callbacks=score_callbacks,
            array_callbacks=array_callbacks,
            plot_callbacks=plot_callbacks,
            score_collection_callbacks=score_collection_callbacks,
            array_collection_callbacks=array_collection_callbacks,
            plot_collection_callbacks=plot_collection_callbacks,
            tracking_client=tracking_client,
        )
        plan = cls(handler_suite=handler_suite, tracking_client=tracking_client)
        return plan

    @staticmethod
    @abstractmethod
    def _get_score_callbacks() -> Sequence[ForwardHookScoreCallback[ModelTContr]]: ...

    @staticmethod
    @abstractmethod
    def _get_array_callbacks() -> Sequence[ForwardHookArrayCallback[ModelTContr]]: ...

    @staticmethod
    @abstractmethod
    def _get_plot_callbacks() -> Sequence[ForwardHookPlotCallback[ModelTContr]]: ...

    @staticmethod
    @abstractmethod
    def _get_score_collection_callbacks() -> Sequence[
        ForwardHookScoreCollectionCallback[ModelTContr]
    ]: ...

    @staticmethod
    @abstractmethod
    def _get_array_collection_callbacks() -> Sequence[
        ForwardHookArrayCollectionCallback[ModelTContr]
    ]: ...

    @staticmethod
    @abstractmethod
    def _get_plot_collection_callbacks() -> Sequence[
        ForwardHookPlotCollectionCallback[ModelTContr]
    ]: ...

    def attach(self, resources: HookCallbackResources[ModelTContr]) -> bool:
        any_attached = False
        for handler in self._ls_handlers:
            any_attached |= handler.attach(resources=resources)
        return any_attached
