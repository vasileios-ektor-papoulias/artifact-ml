from typing import Any, List

from artifact_experiment import DataSplit
from artifact_torch.base.components.callbacks.forward_hook import (
    ForwardHookArray,
    ForwardHookArrayCollection,
    ForwardHookPlot,
    ForwardHookPlotCollection,
    ForwardHookScore,
    ForwardHookScoreCollection,
)
from artifact_torch.base.components.callbacks.loader import (
    DataLoaderArray,
    DataLoaderArrayCollection,
    DataLoaderPlot,
    DataLoaderPlotCollection,
    DataLoaderScore,
    DataLoaderScoreCollection,
)
from artifact_torch.base.components.routines.loader import DataLoaderRoutine
from artifact_torch.base.model.base import Model
from artifact_torch.libs.components.callbacks.forward_hook.activation_pdf import AllActivationsPDF
from artifact_torch.libs.components.callbacks.loader.loss import LoaderLossCallback

from demos.binary_classification.components.routines.protocols import (
    DemoModelInput,
    DemoModelOutput,
)
from demos.binary_classification.config.constants import TRAIN_LOADER_ROUTINE_PERIOD


class DemoLoaderRoutine(
    DataLoaderRoutine[Model[Any, DemoModelOutput], DemoModelInput, DemoModelOutput]
):
    @staticmethod
    def _get_scores(
        data_split: DataSplit,
    ) -> List[DataLoaderScore[DemoModelInput, DemoModelOutput]]:
        return [LoaderLossCallback(period=TRAIN_LOADER_ROUTINE_PERIOD, data_split=data_split)]

    @staticmethod
    def _get_arrays(
        data_split: DataSplit,
    ) -> List[DataLoaderArray[DemoModelInput, DemoModelOutput]]:
        _ = data_split
        return []

    @staticmethod
    def _get_plots(
        data_split: DataSplit,
    ) -> List[DataLoaderPlot[DemoModelInput, DemoModelOutput]]:
        _ = data_split
        return []

    @staticmethod
    def _get_score_collections(
        data_split: DataSplit,
    ) -> List[DataLoaderScoreCollection[DemoModelInput, DemoModelOutput]]:
        _ = data_split
        return []

    @staticmethod
    def _get_array_collections(
        data_split: DataSplit,
    ) -> List[DataLoaderArrayCollection[DemoModelInput, DemoModelOutput]]:
        _ = data_split
        return []

    @staticmethod
    def _get_plot_collections(
        data_split: DataSplit,
    ) -> List[DataLoaderPlotCollection[DemoModelInput, DemoModelOutput]]:
        _ = data_split
        return []

    @staticmethod
    def _get_score_hooks(
        data_split: DataSplit,
    ) -> List[ForwardHookScore[Model[Any, DemoModelOutput]]]:
        _ = data_split
        return []

    @staticmethod
    def _get_array_hooks(
        data_split: DataSplit,
    ) -> List[ForwardHookArray[Model[Any, DemoModelOutput]]]:
        _ = data_split
        return []

    @staticmethod
    def _get_plot_hooks(
        data_split: DataSplit,
    ) -> List[ForwardHookPlot[Model[Any, DemoModelOutput]]]:
        _ = data_split
        return [AllActivationsPDF(period=TRAIN_LOADER_ROUTINE_PERIOD, data_split=data_split)]

    @staticmethod
    def _get_score_collection_hooks(
        data_split: DataSplit,
    ) -> List[ForwardHookScoreCollection[Model[Any, DemoModelOutput]]]:
        _ = data_split
        return []

    @staticmethod
    def _get_array_collection_hooks(
        data_split: DataSplit,
    ) -> List[ForwardHookArrayCollection[Model[Any, DemoModelOutput]]]:
        _ = data_split
        return []

    @staticmethod
    def _get_plot_collection_hooks(
        data_split: DataSplit,
    ) -> List[ForwardHookPlotCollection[Model[Any, DemoModelOutput]]]:
        _ = data_split
        return []
