from typing import Any, List

from artifact_experiment import DataSplit
from artifact_torch.base.components.callbacks.loader_hook import (
    DataLoaderHookArrayCallback,
    DataLoaderHookArrayCollectionCallback,
    DataLoaderHookPlotCallback,
    DataLoaderHookPlotCollectionCallback,
    DataLoaderHookScoreCallback,
    DataLoaderHookScoreCollectionCallback,
)
from artifact_torch.base.components.routines.loader_hook import DataLoaderHookRoutine
from artifact_torch.base.model.base import Model
from artifact_torch.libs.components.callbacks.loader_hook.activation_pdf import (
    AllActivationsPDFCallback,
)

from demos.binary_classification.components.routines.protocols import (
    DemoModelInput,
    DemoModelOutput,
)
from demos.binary_classification.config.constants import TRAIN_LOADER_ROUTINE_PERIOD


class DemoLoaderHookRoutine(
    DataLoaderHookRoutine[Model[Any, DemoModelOutput], DemoModelInput, DemoModelOutput]
):
    @staticmethod
    def _get_score_callbacks(
        data_split: DataSplit,
    ) -> List[
        DataLoaderHookScoreCallback[Model[Any, DemoModelOutput], DemoModelInput, DemoModelOutput]
    ]:
        _ = data_split
        return []

    @staticmethod
    def _get_array_callbacks(
        data_split: DataSplit,
    ) -> List[
        DataLoaderHookArrayCallback[Model[Any, DemoModelOutput], DemoModelInput, DemoModelOutput]
    ]:
        _ = data_split
        return []

    @staticmethod
    def _get_plot_callbacks(
        data_split: DataSplit,
    ) -> List[
        DataLoaderHookPlotCallback[Model[Any, DemoModelOutput], DemoModelInput, DemoModelOutput]
    ]:
        return [
            AllActivationsPDFCallback(period=TRAIN_LOADER_ROUTINE_PERIOD, data_split=data_split)
        ]

    @staticmethod
    def _get_score_collection_callbacks(
        data_split: DataSplit,
    ) -> List[
        DataLoaderHookScoreCollectionCallback[
            Model[Any, DemoModelOutput], DemoModelInput, DemoModelOutput
        ]
    ]:
        _ = data_split
        return []

    @staticmethod
    def _get_array_collection_callbacks(
        data_split: DataSplit,
    ) -> List[
        DataLoaderHookArrayCollectionCallback[
            Model[Any, DemoModelOutput], DemoModelInput, DemoModelOutput
        ]
    ]:
        _ = data_split
        return []

    @staticmethod
    def _get_plot_collection_callbacks(
        data_split: DataSplit,
    ) -> List[
        DataLoaderHookPlotCollectionCallback[
            Model[Any, DemoModelOutput], DemoModelInput, DemoModelOutput
        ]
    ]:
        _ = data_split
        return []
