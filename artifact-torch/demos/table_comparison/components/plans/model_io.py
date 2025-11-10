from typing import List

from artifact_torch.base.components.callbacks.model_io import (
    ModelIOArrayCallback,
    ModelIOArrayCollectionCallback,
    ModelIOPlotCallback,
    ModelIOPlotCollectionCallback,
    ModelIOScoreCallback,
    ModelIOScoreCollectionCallback,
)
from artifact_torch.base.components.plans.model_io import ModelIOPlan, ModelIOPlanBuildContext
from artifact_torch.libs.components.callbacks.loader.loss import LossCallback

from demos.table_comparison.components.protocols import DemoModelInput, DemoModelOutput
from demos.table_comparison.config.constants import TRAIN_LOADER_ROUTINE_PERIOD


class DemoModelIOPlan(ModelIOPlan[DemoModelInput, DemoModelOutput]):
    @classmethod
    def _get_score_callbacks(
        cls, context: ModelIOPlanBuildContext
    ) -> List[ModelIOScoreCallback[DemoModelInput, DemoModelOutput]]:
        return [LossCallback(period=TRAIN_LOADER_ROUTINE_PERIOD, writer=context.score_writer)]

    @classmethod
    def _get_array_callbacks(
        cls, context: ModelIOPlanBuildContext
    ) -> List[ModelIOArrayCallback[DemoModelInput, DemoModelOutput]]:
        _ = context
        return []

    @classmethod
    def _get_plot_callbacks(
        cls, context: ModelIOPlanBuildContext
    ) -> List[ModelIOPlotCallback[DemoModelInput, DemoModelOutput]]:
        _ = context
        return []

    @classmethod
    def _get_score_collection_callbacks(
        cls, context: ModelIOPlanBuildContext
    ) -> List[ModelIOScoreCollectionCallback[DemoModelInput, DemoModelOutput]]:
        _ = context
        return []

    @classmethod
    def _get_array_collection_callbacks(
        cls, context: ModelIOPlanBuildContext
    ) -> List[ModelIOArrayCollectionCallback[DemoModelInput, DemoModelOutput]]:
        _ = context
        return []

    @classmethod
    def _get_plot_collection_callbacks(
        cls, context: ModelIOPlanBuildContext
    ) -> List[ModelIOPlotCollectionCallback[DemoModelInput, DemoModelOutput]]:
        _ = context
        return []
