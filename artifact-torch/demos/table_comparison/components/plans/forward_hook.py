from typing import Any, Sequence

from artifact_torch.base.components.callbacks.forward_hook import (
    ForwardHookArrayCallback,
    ForwardHookArrayCollectionCallback,
    ForwardHookPlotCallback,
    ForwardHookPlotCollectionCallback,
    ForwardHookScoreCallback,
    ForwardHookScoreCollectionCallback,
)
from artifact_torch.base.components.plans.forward_hook import ForwardHookPlan
from artifact_torch.base.model.base import Model
from artifact_torch.libs.components.callbacks.forward_hook.activation_pdf import AllActivationsPDF

from demos.table_comparison.config.constants import TRAIN_LOADER_ROUTINE_PERIOD


class DemoForwardHookPlan(ForwardHookPlan[Model[Any, Any]]):
    @staticmethod
    def _get_score_callbacks() -> Sequence[ForwardHookScoreCallback[Model[Any, Any]]]:
        return []

    @staticmethod
    def _get_array_callbacks() -> Sequence[ForwardHookArrayCallback[Model[Any, Any]]]:
        return []

    @staticmethod
    def _get_plot_callbacks() -> Sequence[ForwardHookPlotCallback[Model[Any, Any]]]:
        return [AllActivationsPDF(period=TRAIN_LOADER_ROUTINE_PERIOD)]

    @staticmethod
    def _get_score_collection_callbacks() -> Sequence[
        ForwardHookScoreCollectionCallback[Model[Any, Any]]
    ]:
        return []

    @staticmethod
    def _get_array_collection_callbacks() -> Sequence[
        ForwardHookArrayCollectionCallback[Model[Any, Any]]
    ]:
        return []

    @staticmethod
    def _get_plot_collection_callbacks() -> Sequence[
        ForwardHookPlotCollectionCallback[Model[Any, Any]]
    ]:
        return []
