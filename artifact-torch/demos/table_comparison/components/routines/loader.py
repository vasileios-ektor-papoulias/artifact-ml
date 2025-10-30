from typing import List

from artifact_experiment import DataSplit
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

from demos.table_comparison.components.routines.protocols import DemoModelInput, DemoModelOutput
from demos.table_comparison.config.constants import TRAIN_LOADER_CALLBACK_PERIOD


class DemoLoaderRoutine(DataLoaderRoutine[DemoModelInput, DemoModelOutput]):
    @staticmethod
    def _get_score_callbacks(
        data_split: DataSplit,
    ) -> List[DataLoaderScoreCallback[DemoModelInput, DemoModelOutput]]:
        return [LoaderLossCallback(period=TRAIN_LOADER_CALLBACK_PERIOD, data_split=data_split)]

    @staticmethod
    def _get_array_callbacks(
        data_split: DataSplit,
    ) -> List[DataLoaderArrayCallback[DemoModelInput, DemoModelOutput]]:
        _ = data_split
        return []

    @staticmethod
    def _get_plot_callbacks(
        data_split: DataSplit,
    ) -> List[DataLoaderPlotCallback[DemoModelInput, DemoModelOutput]]:
        _ = data_split
        return []

    @staticmethod
    def _get_score_collection_callbacks(
        data_split: DataSplit,
    ) -> List[DataLoaderScoreCollectionCallback[DemoModelInput, DemoModelOutput]]:
        _ = data_split
        return []

    @staticmethod
    def _get_array_collection_callbacks(
        data_split: DataSplit,
    ) -> List[DataLoaderArrayCollectionCallback[DemoModelInput, DemoModelOutput]]:
        _ = data_split
        return []

    @staticmethod
    def _get_plot_collection_callbacks(
        data_split: DataSplit,
    ) -> List[DataLoaderPlotCollectionCallback[DemoModelInput, DemoModelOutput]]:
        _ = data_split
        return []
