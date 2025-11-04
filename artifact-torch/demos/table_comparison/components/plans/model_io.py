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

from demos.table_comparison.components.protocols import DemoModelInput, DemoModelOutput
from demos.table_comparison.config.constants import TRAIN_LOADER_ROUTINE_PERIOD


class DemoModelIOPlan(ModelIOPlan[DemoModelInput, DemoModelOutput]):
    @staticmethod
    def _get_score_callbacks() -> List[ModelIOScoreCallback[DemoModelInput, DemoModelOutput]]:
        return [LossCallback(period=TRAIN_LOADER_ROUTINE_PERIOD)]

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
