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
from demos.binary_classification.config.constants import TRAIN_LOADER_CALLBACK_PERIOD
from demos.binary_classification.model.io import MLPClassifierInput, MLPClassifierOutput


class DemoLoaderRoutine(DataLoaderRoutine[MLPClassifierInput, MLPClassifierOutput]):
    @staticmethod
    def _get_score_callbacks() -> List[
        DataLoaderScoreCallback[MLPClassifierInput, MLPClassifierOutput]
    ]:
        return [TrainLossCallback(period=TRAIN_LOADER_CALLBACK_PERIOD)]

    @staticmethod
    def _get_array_callbacks() -> List[
        DataLoaderArrayCallback[MLPClassifierInput, MLPClassifierOutput]
    ]:
        return []

    @staticmethod
    def _get_plot_callbacks() -> List[
        DataLoaderPlotCallback[MLPClassifierInput, MLPClassifierOutput]
    ]:
        return []

    @staticmethod
    def _get_score_collection_callbacks() -> List[
        DataLoaderScoreCollectionCallback[MLPClassifierInput, MLPClassifierOutput]
    ]:
        return []

    @staticmethod
    def _get_array_collection_callbacks() -> List[
        DataLoaderArrayCollectionCallback[MLPClassifierInput, MLPClassifierOutput]
    ]:
        return []

    @staticmethod
    def _get_plot_collection_callbacks() -> List[
        DataLoaderPlotCollectionCallback[MLPClassifierInput, MLPClassifierOutput]
    ]:
        return []
