from typing import List

from artifact_torch.nn.callbacks.model_io import (
    LossCallback,
    ModelIOArrayCallback,
    ModelIOArrayCollectionCallback,
    ModelIOPlotCallback,
    ModelIOPlotCollectionCallback,
    ModelIOScoreCallback,
    ModelIOScoreCollectionCallback,
)
from artifact_torch.nn.plans import ModelIOPlan, ModelIOPlanBuildContext

from demos.table_comparison.config.constants import TRAIN_LOADER_ROUTINE_PERIOD
from demos.table_comparison.contracts.workflow import WorkflowInput, WorkflowOutput


class DemoModelIOPlan(ModelIOPlan[WorkflowInput, WorkflowOutput]):
    @classmethod
    def _get_score_callbacks(
        cls, context: ModelIOPlanBuildContext
    ) -> List[ModelIOScoreCallback[WorkflowInput, WorkflowOutput]]:
        return [LossCallback(period=TRAIN_LOADER_ROUTINE_PERIOD, writer=context.score_writer)]

    @classmethod
    def _get_array_callbacks(
        cls, context: ModelIOPlanBuildContext
    ) -> List[ModelIOArrayCallback[WorkflowInput, WorkflowOutput]]:
        _ = context
        return []

    @classmethod
    def _get_plot_callbacks(
        cls, context: ModelIOPlanBuildContext
    ) -> List[ModelIOPlotCallback[WorkflowInput, WorkflowOutput]]:
        _ = context
        return []

    @classmethod
    def _get_score_collection_callbacks(
        cls, context: ModelIOPlanBuildContext
    ) -> List[ModelIOScoreCollectionCallback[WorkflowInput, WorkflowOutput]]:
        _ = context
        return []

    @classmethod
    def _get_array_collection_callbacks(
        cls, context: ModelIOPlanBuildContext
    ) -> List[ModelIOArrayCollectionCallback[WorkflowInput, WorkflowOutput]]:
        _ = context
        return []

    @classmethod
    def _get_plot_collection_callbacks(
        cls, context: ModelIOPlanBuildContext
    ) -> List[ModelIOPlotCollectionCallback[WorkflowInput, WorkflowOutput]]:
        _ = context
        return []
