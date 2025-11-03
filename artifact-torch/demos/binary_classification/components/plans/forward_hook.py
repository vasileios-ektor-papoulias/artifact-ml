from typing import Any, Sequence

from artifact_experiment import DataSplit
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

from demos.binary_classification.config.constants import TRAIN_LOADER_ROUTINE_PERIOD


class DemoForwardHookPlan(ForwardHookPlan[Model[Any, Any]]):
    @staticmethod
    def _get_score_callbacks(
        data_split: DataSplit,
    ) -> Sequence[ForwardHookScoreCallback[Model[Any, Any]]]:
        _ = data_split
        return []

    @staticmethod
    def _get_array_callbacks(
        data_split: DataSplit,
    ) -> Sequence[ForwardHookArrayCallback[Model[Any, Any]]]:
        _ = data_split
        return []

    @staticmethod
    def _get_plot_callbacks(
        data_split: DataSplit,
    ) -> Sequence[ForwardHookPlotCallback[Model[Any, Any]]]:
        _ = data_split
        return [AllActivationsPDF(period=TRAIN_LOADER_ROUTINE_PERIOD, data_split=data_split)]

    @staticmethod
    def _get_score_collection_callbacks(
        data_split: DataSplit,
    ) -> Sequence[ForwardHookScoreCollectionCallback[Model[Any, Any]]]:
        _ = data_split
        return []

    @staticmethod
    def _get_array_collection_callbacks(
        data_split: DataSplit,
    ) -> Sequence[ForwardHookArrayCollectionCallback[Model[Any, Any]]]:
        _ = data_split
        return []

    @staticmethod
    def _get_plot_collection_callbacks(
        data_split: DataSplit,
    ) -> Sequence[ForwardHookPlotCollectionCallback[Model[Any, Any]]]:
        _ = data_split
        return []
