from abc import abstractmethod
from typing import Any, Generic, Optional, Sequence, Type, TypeVar

from artifact_experiment.base.entities.data_split import DataSplit
from artifact_experiment.base.plans.base import CallbackExecutionPlan
from artifact_experiment.base.tracking.client import TrackingClient

from artifact_torch.base.components.callbacks.forward_hook import (
    ForwardHookArrayCallback,
    ForwardHookArrayCollectionCallback,
    ForwardHookCallback,
    ForwardHookCallbackHandler,
    ForwardHookCallbackResources,
    ForwardHookHandlerSuite,
    ForwardHookPlotCallback,
    ForwardHookPlotCollectionCallback,
    ForwardHookScoreCallback,
    ForwardHookScoreCollectionCallback,
)
from artifact_torch.base.model.base import Model

ModelTContr = TypeVar("ModelTContr", bound=Model[Any, Any], contravariant=True)
ForwardHookPlanT = TypeVar("ForwardHookPlanT", bound="ForwardHookPlan")


class ForwardHookPlan(
    CallbackExecutionPlan[
        ForwardHookCallbackHandler[ForwardHookCallback[ModelTContr, Any, Any], ModelTContr, Any],
        ForwardHookCallback[ModelTContr, Any, Any],
        ForwardHookCallbackResources[ModelTContr],
    ],
    Generic[ModelTContr],
):
    def __init__(
        self,
        callback_handlers: ForwardHookHandlerSuite[ModelTContr],
        data_split: Optional[DataSplit] = None,
        tracking_client: Optional[TrackingClient] = None,
    ):
        super().__init__(
            callback_handlers=callback_handlers,
            data_split=data_split,
            tracking_client=tracking_client,
        )

    @classmethod
    def build(
        cls: Type[ForwardHookPlanT],
        data_split: DataSplit,
        tracking_client: Optional[TrackingClient] = None,
    ) -> ForwardHookPlanT:
        score_callbacks = cls._get_score_callbacks(data_split=data_split)
        array_callbacks = cls._get_array_callbacks(data_split=data_split)
        plot_callbacks = cls._get_plot_callbacks(data_split=data_split)
        score_collection_callbacks = cls._get_score_collection_callbacks(data_split=data_split)
        array_collection_callbacks = cls._get_array_collection_callbacks(data_split=data_split)
        plot_collection_callbacks = cls._get_plot_collection_callbacks(data_split=data_split)
        callback_handlers = ForwardHookHandlerSuite[ModelTContr].build(
            score_callbacks=score_callbacks,
            array_callbacks=array_callbacks,
            plot_callbacks=plot_callbacks,
            score_collection_callbacks=score_collection_callbacks,
            array_collection_callbacks=array_collection_callbacks,
            plot_collection_callbacks=plot_collection_callbacks,
            tracking_client=tracking_client,
        )
        plan = cls(
            callback_handlers=callback_handlers,
            data_split=data_split,
            tracking_client=tracking_client,
        )
        return plan

    @staticmethod
    @abstractmethod
    def _get_score_callbacks(
        data_split: DataSplit,
    ) -> Sequence[ForwardHookScoreCallback[ModelTContr]]: ...

    @staticmethod
    @abstractmethod
    def _get_array_callbacks(
        data_split: DataSplit,
    ) -> Sequence[ForwardHookArrayCallback[ModelTContr]]: ...

    @staticmethod
    @abstractmethod
    def _get_plot_callbacks(
        data_split: DataSplit,
    ) -> Sequence[ForwardHookPlotCallback[ModelTContr]]: ...

    @staticmethod
    @abstractmethod
    def _get_score_collection_callbacks(
        data_split: DataSplit,
    ) -> Sequence[ForwardHookScoreCollectionCallback[ModelTContr]]: ...

    @staticmethod
    @abstractmethod
    def _get_array_collection_callbacks(
        data_split: DataSplit,
    ) -> Sequence[ForwardHookArrayCollectionCallback[ModelTContr]]: ...

    @staticmethod
    @abstractmethod
    def _get_plot_collection_callbacks(
        data_split: DataSplit,
    ) -> Sequence[ForwardHookPlotCollectionCallback[ModelTContr]]: ...

    def attach(self, resources: ForwardHookCallbackResources[ModelTContr]) -> bool:
        any_attached = False
        for handler in self._ls_handlers:
            any_attached |= handler.attach(resources=resources)
        return any_attached
