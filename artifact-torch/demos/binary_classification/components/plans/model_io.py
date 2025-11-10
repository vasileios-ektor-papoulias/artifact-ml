from typing import List

from artifact_torch.base.components.callbacks.model_io import (
    ModelIOArrayCallback,
    ModelIOArrayCollectionCallback,
    ModelIOPlotCallback,
    ModelIOPlotCollectionCallback,
    ModelIOScoreCallback,
    ModelIOScoreCollectionCallback,
)
from artifact_torch.base.components.plans.model_io import ModelIOPlan, ModelIOPlanBuildContext
from artifact_torch.libs.components.callbacks.loader.loss import LossCallback

from demos.binary_classification.components.protocols import DemoModelInput, DemoModelOutput
from demos.binary_classification.config.constants import (
    LOADER_VALIDATION_PERIOD,
    TRAIN_DIAGNOSTICS_PERIOD,
)


class DataLoaderModelIOPlan(ModelIOPlan[DemoModelInput, DemoModelOutput]):
    @classmethod
    def _get_score_callbacks(
        cls, context: ModelIOPlanBuildContext
    ) -> List[ModelIOScoreCallback[DemoModelInput, DemoModelOutput]]:
        return [LossCallback(period=LOADER_VALIDATION_PERIOD, writer=context.score_writer)]

    @classmethod
    def _get_array_callbacks(
        cls, context: ModelIOPlanBuildContext
    ) -> List[ModelIOArrayCallback[DemoModelInput, DemoModelOutput]]:
        _ = context
        return []

    @classmethod
    def _get_plot_callbacks(
        cls, context: ModelIOPlanBuildContext
    ) -> List[ModelIOPlotCallback[DemoModelInput, DemoModelOutput]]:
        _ = context
        return []

    @classmethod
    def _get_score_collection_callbacks(
        cls, context: ModelIOPlanBuildContext
    ) -> List[ModelIOScoreCollectionCallback[DemoModelInput, DemoModelOutput]]:
        _ = context
        return []

    @classmethod
    def _get_array_collection_callbacks(
        cls, context: ModelIOPlanBuildContext
    ) -> List[ModelIOArrayCollectionCallback[DemoModelInput, DemoModelOutput]]:
        _ = context
        return []

    @classmethod
    def _get_plot_collection_callbacks(
        cls, context: ModelIOPlanBuildContext
    ) -> List[ModelIOPlotCollectionCallback[DemoModelInput, DemoModelOutput]]:
        _ = context
        return []


class TrainDiagnosticsModelIOPlan(ModelIOPlan[DemoModelInput, DemoModelOutput]):
    @classmethod
    def _get_score_callbacks(
        cls, context: ModelIOPlanBuildContext
    ) -> List[ModelIOScoreCallback[DemoModelInput, DemoModelOutput]]:
        return [LossCallback(period=TRAIN_DIAGNOSTICS_PERIOD, writer=context.score_writer)]

    @classmethod
    def _get_array_callbacks(
        cls, context: ModelIOPlanBuildContext
    ) -> List[ModelIOArrayCallback[DemoModelInput, DemoModelOutput]]:
        _ = context
        return []

    @classmethod
    def _get_plot_callbacks(
        cls, context: ModelIOPlanBuildContext
    ) -> List[ModelIOPlotCallback[DemoModelInput, DemoModelOutput]]:
        _ = context
        return []

    @classmethod
    def _get_score_collection_callbacks(
        cls, context: ModelIOPlanBuildContext
    ) -> List[ModelIOScoreCollectionCallback[DemoModelInput, DemoModelOutput]]:
        _ = context
        return []

    @classmethod
    def _get_array_collection_callbacks(
        cls, context: ModelIOPlanBuildContext
    ) -> List[ModelIOArrayCollectionCallback[DemoModelInput, DemoModelOutput]]:
        _ = context
        return []

    @classmethod
    def _get_plot_collection_callbacks(
        cls, context: ModelIOPlanBuildContext
    ) -> List[ModelIOPlotCollectionCallback[DemoModelInput, DemoModelOutput]]:
        _ = context
        return []
