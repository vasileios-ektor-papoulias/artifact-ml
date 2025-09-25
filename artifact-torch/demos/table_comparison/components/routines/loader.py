from typing import List

from artifact_torch.base.components.callbacks.data_loader import (
    DataLoaderArrayCallback,
    DataLoaderArrayCollectionCallback,
    DataLoaderPlotCallback,
    DataLoaderPlotCollectionCallback,
    DataLoaderScoreCallback,
    DataLoaderScoreCollectionCallback,
)
from artifact_torch.base.components.routines.data_loader import DataLoaderRoutine
from artifact_torch.libs.components.callbacks.data_loader.loss import TrainLossCallback
from demos.table_comparison.config.constants import TRAIN_LOADER_CALLBACK_PERIOD
from demos.table_comparison.model.io import TabularVAEInput, TabularVAEOutput


class DemoLoaderRoutine(DataLoaderRoutine[TabularVAEInput, TabularVAEOutput]):
    @staticmethod
    def _get_score_callbacks() -> List[DataLoaderScoreCallback[TabularVAEInput, TabularVAEOutput]]:
        return [TrainLossCallback(period=TRAIN_LOADER_CALLBACK_PERIOD)]

    @staticmethod
    def _get_array_callbacks() -> List[DataLoaderArrayCallback[TabularVAEInput, TabularVAEOutput]]:
        return []

    @staticmethod
    def _get_plot_callbacks() -> List[DataLoaderPlotCallback[TabularVAEInput, TabularVAEOutput]]:
        return []

    @staticmethod
    def _get_score_collection_callbacks() -> List[
        DataLoaderScoreCollectionCallback[TabularVAEInput, TabularVAEOutput]
    ]:
        return []

    @staticmethod
    def _get_array_collection_callbacks() -> List[
        DataLoaderArrayCollectionCallback[TabularVAEInput, TabularVAEOutput]
    ]:
        return []

    @staticmethod
    def _get_plot_collection_callbacks() -> List[
        DataLoaderPlotCollectionCallback[TabularVAEInput, TabularVAEOutput]
    ]:
        return []
