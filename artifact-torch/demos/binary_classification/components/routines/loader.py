from typing import List

from artifact_experiment.base.data_split import DataSplit
from artifact_torch.base.components.callbacks.loader import (
    DataLoaderArrayCallback,
    DataLoaderArrayCollectionCallback,
    DataLoaderPlotCallback,
    DataLoaderPlotCollectionCallback,
    DataLoaderScoreCallback,
    DataLoaderScoreCollectionCallback,
)
from artifact_torch.base.components.routines.loader import DataLoaderRoutine
from artifact_torch.libs.components.callbacks.loader.loss import LoaderLossCallback

from demos.binary_classification.config.constants import TRAIN_LOADER_CALLBACK_PERIOD
from demos.binary_classification.model.io import MLPClassifierInput, MLPClassifierOutput


class DemoLoaderRoutine(DataLoaderRoutine[MLPClassifierInput, MLPClassifierOutput]):
    @staticmethod
    def _get_score_callbacks(
        data_split: DataSplit,
    ) -> List[DataLoaderScoreCallback[MLPClassifierInput, MLPClassifierOutput]]:
        return [LoaderLossCallback(period=TRAIN_LOADER_CALLBACK_PERIOD, data_split=data_split)]

    @staticmethod
    def _get_array_callbacks(
        data_split: DataSplit,
    ) -> List[DataLoaderArrayCallback[MLPClassifierInput, MLPClassifierOutput]]:
        _ = data_split
        return []

    @staticmethod
    def _get_plot_callbacks(
        data_split: DataSplit,
    ) -> List[DataLoaderPlotCallback[MLPClassifierInput, MLPClassifierOutput]]:
        _ = data_split
        return []

    @staticmethod
    def _get_score_collection_callbacks(
        data_split: DataSplit,
    ) -> List[DataLoaderScoreCollectionCallback[MLPClassifierInput, MLPClassifierOutput]]:
        _ = data_split
        return []

    @staticmethod
    def _get_array_collection_callbacks(
        data_split: DataSplit,
    ) -> List[DataLoaderArrayCollectionCallback[MLPClassifierInput, MLPClassifierOutput]]:
        _ = data_split
        return []

    @staticmethod
    def _get_plot_collection_callbacks(
        data_split: DataSplit,
    ) -> List[DataLoaderPlotCollectionCallback[MLPClassifierInput, MLPClassifierOutput]]:
        _ = data_split
        return []
