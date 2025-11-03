from typing import List

from artifact_experiment import DataSplit
from artifact_torch.base.components.callbacks.model_io import (
    ModelIOArrayCallback,
    ModelIOArrayCollectionCallback,
    ModelIOPlotCallback,
    ModelIOPlotCollectionCallback,
    ModelIOScoreCallback,
    ModelIOScoreCollectionCallback,
)
from artifact_torch.base.components.plans.model_io import ModelIOPlan
from artifact_torch.libs.components.callbacks.loader.loss import LoaderLossCallback

from demos.binary_classification.components.protocols import DemoModelInput, DemoModelOutput
from demos.binary_classification.config.constants import TRAIN_LOADER_ROUTINE_PERIOD


class DemoModelIOPlan(ModelIOPlan[DemoModelInput, DemoModelOutput]):
    @staticmethod
    def _get_score_callbacks(
        data_split: DataSplit,
    ) -> List[ModelIOScoreCallback[DemoModelInput, DemoModelOutput]]:
        return [LoaderLossCallback(period=TRAIN_LOADER_ROUTINE_PERIOD, data_split=data_split)]

    @staticmethod
    def _get_array_callbacks(
        data_split: DataSplit,
    ) -> List[ModelIOArrayCallback[DemoModelInput, DemoModelOutput]]:
        _ = data_split
        return []

    @staticmethod
    def _get_plot_callbacks(
        data_split: DataSplit,
    ) -> List[ModelIOPlotCallback[DemoModelInput, DemoModelOutput]]:
        _ = data_split
        return []

    @staticmethod
    def _get_score_collection_callbacks(
        data_split: DataSplit,
    ) -> List[ModelIOScoreCollectionCallback[DemoModelInput, DemoModelOutput]]:
        _ = data_split
        return []

    @staticmethod
    def _get_array_collection_callbacks(
        data_split: DataSplit,
    ) -> List[ModelIOArrayCollectionCallback[DemoModelInput, DemoModelOutput]]:
        _ = data_split
        return []

    @staticmethod
    def _get_plot_collection_callbacks(
        data_split: DataSplit,
    ) -> List[ModelIOPlotCollectionCallback[DemoModelInput, DemoModelOutput]]:
        _ = data_split
        return []
