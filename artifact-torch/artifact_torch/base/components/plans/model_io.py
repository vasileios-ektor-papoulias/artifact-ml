from abc import abstractmethod
from typing import Any, Generic, Optional, Sequence, Type, TypeVar

from artifact_experiment.base.plans.base import CallbackExecutionPlan
from artifact_experiment.base.tracking.client import TrackingClient

from artifact_torch.base.components.callbacks.model_io import (
    ModelIOArrayCallback,
    ModelIOArrayCollectionCallback,
    ModelIOCallback,
    ModelIOCallbackResources,
    ModelIOPlotCallback,
    ModelIOPlotCollectionCallback,
    ModelIOScoreCallback,
    ModelIOScoreCollectionCallback,
)
from artifact_torch.base.components.handlers.model_io import (
    ModelIOCallbackHandler,
    ModelIOCallbackHandlerSuite,
)
from artifact_torch.base.model.io import ModelInput, ModelOutput

ModelInputTContr = TypeVar("ModelInputTContr", bound=ModelInput, contravariant=True)
ModelOutputTContr = TypeVar("ModelOutputTContr", bound=ModelOutput, contravariant=True)
ModelIOPlanT = TypeVar("ModelIOPlanT", bound="ModelIOPlan[Any, Any]")


class ModelIOPlan(
    CallbackExecutionPlan[
        ModelIOCallbackHandler[ModelOutputTContr, Any],
        ModelIOCallback[ModelInputTContr, ModelOutputTContr, Any, Any],
        ModelIOCallbackResources[ModelOutputTContr],
    ],
    Generic[ModelInputTContr, ModelOutputTContr],
):
    def __init__(
        self,
        handler_suite: ModelIOCallbackHandlerSuite[ModelInputTContr, ModelOutputTContr],
        tracking_client: Optional[TrackingClient] = None,
    ):
        super().__init__(handler_suite=handler_suite, tracking_client=tracking_client)

    @classmethod
    def build(
        cls: Type[ModelIOPlanT], tracking_client: Optional[TrackingClient] = None
    ) -> ModelIOPlanT:
        score_callbacks = cls._get_score_callbacks()
        array_callbacks = cls._get_array_callbacks()
        plot_callbacks = cls._get_plot_callbacks()
        score_collection_callbacks = cls._get_score_collection_callbacks()
        array_collection_callbacks = cls._get_array_collection_callbacks()
        plot_collection_callbacks = cls._get_plot_collection_callbacks()
        handler_suite = ModelIOCallbackHandlerSuite[ModelInputTContr, ModelOutputTContr].build(
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
    def _get_score_callbacks() -> Sequence[
        ModelIOScoreCallback[ModelInputTContr, ModelOutputTContr]
    ]: ...

    @staticmethod
    @abstractmethod
    def _get_array_callbacks() -> Sequence[
        ModelIOArrayCallback[ModelInputTContr, ModelOutputTContr]
    ]: ...

    @staticmethod
    @abstractmethod
    def _get_plot_callbacks() -> Sequence[
        ModelIOPlotCallback[ModelInputTContr, ModelOutputTContr]
    ]: ...

    @staticmethod
    @abstractmethod
    def _get_score_collection_callbacks() -> Sequence[
        ModelIOScoreCollectionCallback[ModelInputTContr, ModelOutputTContr]
    ]: ...

    @staticmethod
    @abstractmethod
    def _get_array_collection_callbacks() -> Sequence[
        ModelIOArrayCollectionCallback[ModelInputTContr, ModelOutputTContr]
    ]: ...

    @staticmethod
    @abstractmethod
    def _get_plot_collection_callbacks() -> Sequence[
        ModelIOPlotCollectionCallback[ModelInputTContr, ModelOutputTContr]
    ]: ...

    def attach(self, resources: ModelIOCallbackResources[ModelOutputTContr]) -> bool:
        any_attached = False
        for handler in self._ls_handlers:
            any_attached |= handler.attach(resources=resources)
        return any_attached
