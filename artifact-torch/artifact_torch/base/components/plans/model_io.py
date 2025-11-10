from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, Sequence, Type, TypeVar

from artifact_core.base.artifact_dependencies import ArtifactResult
from artifact_experiment.base.components.plans.base import CallbackExecutionPlan, PlanBuildContext

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
from artifact_torch.base.components.handler_suites.model_io import ModelIOCallbackHandlerSuite
from artifact_torch.base.model.io import ModelInput, ModelOutput


@dataclass(frozen=True)
class ModelIOPlanBuildContext(PlanBuildContext):
    pass


ModelInputTContr = TypeVar("ModelInputTContr", bound=ModelInput, contravariant=True)
ModelOutputTContr = TypeVar("ModelOutputTContr", bound=ModelOutput, contravariant=True)
ModelIOPlanT = TypeVar("ModelIOPlanT", bound="ModelIOPlan[Any, Any]")


class ModelIOPlan(
    CallbackExecutionPlan[
        ModelIOCallbackHandlerSuite[ModelInputTContr, ModelOutputTContr],
        ModelIOCallback[ModelInputTContr, ModelOutputTContr, ArtifactResult, Any],
        ModelIOCallbackResources[ModelOutputTContr],
        ModelIOPlanBuildContext,
    ],
    Generic[ModelInputTContr, ModelOutputTContr],
):
    @classmethod
    @abstractmethod
    def _get_score_callbacks(
        cls, context: ModelIOPlanBuildContext
    ) -> Sequence[ModelIOScoreCallback[ModelInputTContr, ModelOutputTContr]]: ...

    @classmethod
    @abstractmethod
    def _get_array_callbacks(
        cls, context: ModelIOPlanBuildContext
    ) -> Sequence[ModelIOArrayCallback[ModelInputTContr, ModelOutputTContr]]: ...

    @classmethod
    @abstractmethod
    def _get_plot_callbacks(
        cls, context: ModelIOPlanBuildContext
    ) -> Sequence[ModelIOPlotCallback[ModelInputTContr, ModelOutputTContr]]: ...

    @classmethod
    @abstractmethod
    def _get_score_collection_callbacks(
        cls, context: ModelIOPlanBuildContext
    ) -> Sequence[ModelIOScoreCollectionCallback[ModelInputTContr, ModelOutputTContr]]: ...

    @classmethod
    @abstractmethod
    def _get_array_collection_callbacks(
        cls, context: ModelIOPlanBuildContext
    ) -> Sequence[ModelIOArrayCollectionCallback[ModelInputTContr, ModelOutputTContr]]: ...

    @classmethod
    @abstractmethod
    def _get_plot_collection_callbacks(
        cls, context: ModelIOPlanBuildContext
    ) -> Sequence[ModelIOPlotCollectionCallback[ModelInputTContr, ModelOutputTContr]]: ...

    def attach(self, resources: ModelIOCallbackResources[ModelOutputTContr]) -> bool:
        return self._handler_suite.attach(resources=resources)

    @classmethod
    def _get_handler_suite_type(
        cls,
    ) -> Type[ModelIOCallbackHandlerSuite[ModelInputTContr, ModelOutputTContr]]:
        return ModelIOCallbackHandlerSuite
