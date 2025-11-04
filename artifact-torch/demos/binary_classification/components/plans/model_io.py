from typing import List

from artifact_torch.base.components.callbacks.model_io import (
    ModelIOArrayCallback,
    ModelIOArrayCollectionCallback,
    ModelIOPlotCallback,
    ModelIOPlotCollectionCallback,
    ModelIOScoreCallback,
    ModelIOScoreCollectionCallback,
)
from artifact_torch.base.components.plans.model_io import ModelIOPlan
from artifact_torch.libs.components.callbacks.loader.loss import LossCallback

from demos.binary_classification.components.protocols import DemoModelInput, DemoModelOutput
from demos.binary_classification.config.constants import (
    LOADER_VALIDATION_PERIOD,
    TRAIN_DIAGNOSTICS_PERIOD,
)


class DataLoaderModelIOPlan(ModelIOPlan[DemoModelInput, DemoModelOutput]):
    @staticmethod
    def _get_score_callbacks() -> List[ModelIOScoreCallback[DemoModelInput, DemoModelOutput]]:
        return [LossCallback(period=LOADER_VALIDATION_PERIOD)]

    @staticmethod
    def _get_array_callbacks() -> List[ModelIOArrayCallback[DemoModelInput, DemoModelOutput]]:
        return []

    @staticmethod
    def _get_plot_callbacks() -> List[ModelIOPlotCallback[DemoModelInput, DemoModelOutput]]:
        return []

    @staticmethod
    def _get_score_collection_callbacks() -> List[
        ModelIOScoreCollectionCallback[DemoModelInput, DemoModelOutput]
    ]:
        return []

    @staticmethod
    def _get_array_collection_callbacks() -> List[
        ModelIOArrayCollectionCallback[DemoModelInput, DemoModelOutput]
    ]:
        return []

    @staticmethod
    def _get_plot_collection_callbacks() -> List[
        ModelIOPlotCollectionCallback[DemoModelInput, DemoModelOutput]
    ]:
        return []


class TrainDiagnosticsModelIOPlan(ModelIOPlan[DemoModelInput, DemoModelOutput]):
    @staticmethod
    def _get_score_callbacks() -> List[ModelIOScoreCallback[DemoModelInput, DemoModelOutput]]:
        return [LossCallback(period=TRAIN_DIAGNOSTICS_PERIOD)]

    @staticmethod
    def _get_array_callbacks() -> List[ModelIOArrayCallback[DemoModelInput, DemoModelOutput]]:
        return []

    @staticmethod
    def _get_plot_callbacks() -> List[ModelIOPlotCallback[DemoModelInput, DemoModelOutput]]:
        return []

    @staticmethod
    def _get_score_collection_callbacks() -> List[
        ModelIOScoreCollectionCallback[DemoModelInput, DemoModelOutput]
    ]:
        return []

    @staticmethod
    def _get_array_collection_callbacks() -> List[
        ModelIOArrayCollectionCallback[DemoModelInput, DemoModelOutput]
    ]:
        return []

    @staticmethod
    def _get_plot_collection_callbacks() -> List[
        ModelIOPlotCollectionCallback[DemoModelInput, DemoModelOutput]
    ]:
        return []
