from abc import abstractmethod
from typing import Any, Generic, Optional, Sequence, Type, TypeVar

from artifact_experiment.base.entities.data_split import DataSplit
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
    ModelIOHandlerSuite,
)
from artifact_torch.base.model.io import ModelInput, ModelOutput

ModelInputTContr = TypeVar("ModelInputTContr", bound=ModelInput, contravariant=True)
ModelOutputTContr = TypeVar("ModelOutputTContr", bound=ModelOutput, contravariant=True)
ModelIOPlanT = TypeVar("ModelIOPlanT", bound="ModelIOPlan")


class ModelIOPlan(
    CallbackExecutionPlan[
        ModelIOCallbackHandler[
            ModelInputTContr,
            ModelOutputTContr,
            Any,
        ],
        ModelIOCallback[ModelInputTContr, ModelOutputTContr, Any, Any],
        ModelIOCallbackResources[ModelOutputTContr],
    ],
    Generic[ModelInputTContr, ModelOutputTContr],
):
    def __init__(
        self,
        callback_handlers: ModelIOHandlerSuite[ModelInputTContr, ModelOutputTContr],
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
        cls: Type[ModelIOPlanT],
        data_split: DataSplit,
        tracking_client: Optional[TrackingClient] = None,
    ) -> ModelIOPlanT:
        score_callbacks = cls._get_score_callbacks(data_split=data_split)
        array_callbacks = cls._get_array_callbacks(data_split=data_split)
        plot_callbacks = cls._get_plot_callbacks(data_split=data_split)
        score_collection_callbacks = cls._get_score_collection_callbacks(data_split=data_split)
        array_collection_callbacks = cls._get_array_collection_callbacks(data_split=data_split)
        plot_collection_callbacks = cls._get_plot_collection_callbacks(data_split=data_split)
        callback_handlers = ModelIOHandlerSuite[ModelInputTContr, ModelOutputTContr].build(
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
    ) -> Sequence[ModelIOScoreCallback[ModelInputTContr, ModelOutputTContr]]: ...

    @staticmethod
    @abstractmethod
    def _get_array_callbacks(
        data_split: DataSplit,
    ) -> Sequence[ModelIOArrayCallback[ModelInputTContr, ModelOutputTContr]]: ...

    @staticmethod
    @abstractmethod
    def _get_plot_callbacks(
        data_split: DataSplit,
    ) -> Sequence[ModelIOPlotCallback[ModelInputTContr, ModelOutputTContr]]: ...

    @staticmethod
    @abstractmethod
    def _get_score_collection_callbacks(
        data_split: DataSplit,
    ) -> Sequence[ModelIOScoreCollectionCallback[ModelInputTContr, ModelOutputTContr]]: ...

    @staticmethod
    @abstractmethod
    def _get_array_collection_callbacks(
        data_split: DataSplit,
    ) -> Sequence[ModelIOArrayCollectionCallback[ModelInputTContr, ModelOutputTContr]]: ...

    @staticmethod
    @abstractmethod
    def _get_plot_collection_callbacks(
        data_split: DataSplit,
    ) -> Sequence[ModelIOPlotCollectionCallback[ModelInputTContr, ModelOutputTContr]]: ...

    def attach(self, resources: ModelIOCallbackResources[ModelOutputTContr]) -> bool:
        any_attached = False
        for handler in self._ls_handlers:
            any_attached |= handler.attach(resources=resources)
        return any_attached
