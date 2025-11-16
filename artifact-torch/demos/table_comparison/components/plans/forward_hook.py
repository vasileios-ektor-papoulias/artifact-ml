from typing import Any, Sequence

from artifact_torch.nn import Model
from artifact_torch.nn.callbacks.forward_hook import (
    AllActivationsPDF,
    ForwardHookArrayCallback,
    ForwardHookArrayCollectionCallback,
    ForwardHookPlotCallback,
    ForwardHookPlotCollectionCallback,
    ForwardHookScoreCallback,
    ForwardHookScoreCollectionCallback,
)
from artifact_torch.nn.plans import ForwardHookPlan, ForwardHookPlanBuildContext

from demos.table_comparison.config.constants import TRAIN_LOADER_ROUTINE_PERIOD


class DemoForwardHookPlan(ForwardHookPlan[Model[Any, Any]]):
    @classmethod
    def _get_score_callbacks(
        cls, context: ForwardHookPlanBuildContext
    ) -> Sequence[ForwardHookScoreCallback[Model[Any, Any]]]:
        _ = context
        return []

    @classmethod
    def _get_array_callbacks(
        cls, context: ForwardHookPlanBuildContext
    ) -> Sequence[ForwardHookArrayCallback[Model[Any, Any]]]:
        _ = context
        return []

    @classmethod
    def _get_plot_callbacks(
        cls, context: ForwardHookPlanBuildContext
    ) -> Sequence[ForwardHookPlotCallback[Model[Any, Any]]]:
        return [AllActivationsPDF(period=TRAIN_LOADER_ROUTINE_PERIOD, writer=context.plot_writer)]

    @classmethod
    def _get_score_collection_callbacks(
        cls, context: ForwardHookPlanBuildContext
    ) -> Sequence[ForwardHookScoreCollectionCallback[Model[Any, Any]]]:
        _ = context
        return []

    @classmethod
    def _get_array_collection_callbacks(
        cls, context: ForwardHookPlanBuildContext
    ) -> Sequence[ForwardHookArrayCollectionCallback[Model[Any, Any]]]:
        _ = context
        return []

    @classmethod
    def _get_plot_collection_callbacks(
        cls, context: ForwardHookPlanBuildContext
    ) -> Sequence[ForwardHookPlotCollectionCallback[Model[Any, Any]]]:
        _ = context
        return []
