from typing import List, Optional

from artifact_core.libs.resource_spec.tabular.protocol import TabularDataSpecProtocol
from artifact_experiment.base.tracking.client import TrackingClient
from artifact_experiment.table_comparison.validation_plan import (
    TableComparisonArrayCollectionType,
    TableComparisonArrayType,
    TableComparisonPlan,
    TableComparisonPlotCollectionType,
    TableComparisonPlotType,
    TableComparisonScoreCollectionType,
    TableComparisonScoreType,
)
from artifact_torch.table_comparison.routine import TableComparisonRoutine

from demo.config.constants import (
    ARTIFACT_VALIDATION_PERIOD,
    GENERATION_N_RECORDS,
    GENERATION_TEMPERATURE,
    GENERATION_USE_MEAN,
)
from demo.model.synthesizer import TabularVAEGenerationParams


class DemoTableComparisonPlan(TableComparisonPlan):
    @staticmethod
    def _get_score_types() -> List[TableComparisonScoreType]:
        return [
            TableComparisonScoreType.MEAN_JS_DISTANCE,
            TableComparisonScoreType.CORRELATION_DISTANCE,
        ]

    @staticmethod
    def _get_array_types() -> List[TableComparisonArrayType]:
        return []

    @staticmethod
    def _get_plot_types() -> List[TableComparisonPlotType]:
        return [
            TableComparisonPlotType.PDF,
            TableComparisonPlotType.CDF,
            TableComparisonPlotType.DESCRIPTIVE_STATS_ALIGNMENT,
            TableComparisonPlotType.PCA_JUXTAPOSITION,
            TableComparisonPlotType.CORRELATION_HEATMAP_JUXTAPOSITION,
        ]

    @staticmethod
    def _get_score_collection_types() -> List[TableComparisonScoreCollectionType]:
        return [
            TableComparisonScoreCollectionType.JS_DISTANCE,
        ]

    @staticmethod
    def _get_array_collection_types() -> List[TableComparisonArrayCollectionType]:
        return [
            TableComparisonArrayCollectionType.MEAN_JUXTAPOSITION,
            TableComparisonArrayCollectionType.STD_JUXTAPOSITION,
            TableComparisonArrayCollectionType.MIN_JUXTAPOSITION,
            TableComparisonArrayCollectionType.MAX_JUXTAPOSITION,
        ]

    @staticmethod
    def _get_plot_collection_types() -> List[TableComparisonPlotCollectionType]:
        return [
            TableComparisonPlotCollectionType.PDF,
            TableComparisonPlotCollectionType.CDF,
        ]


class DemoTableComparisonRoutine(TableComparisonRoutine[TabularVAEGenerationParams]):
    @classmethod
    def _get_period(cls) -> int:
        return ARTIFACT_VALIDATION_PERIOD

    @classmethod
    def _get_generation_params(cls) -> TabularVAEGenerationParams:
        return TabularVAEGenerationParams(
            n_records=GENERATION_N_RECORDS,
            use_mean=GENERATION_USE_MEAN,
            temperature=GENERATION_TEMPERATURE,
            sample=True,
        )

    @classmethod
    def _get_validation_plan(
        cls,
        artifact_resource_spec: TabularDataSpecProtocol,
        tracking_client: Optional[TrackingClient],
    ) -> TableComparisonPlan:
        return DemoTableComparisonPlan.build(
            resource_spec=artifact_resource_spec, tracking_client=tracking_client
        )
