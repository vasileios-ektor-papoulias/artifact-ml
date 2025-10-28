from typing import List, Mapping, Optional

from artifact_core.table_comparison import (
    TableComparisonArrayCollectionType,
    TableComparisonArrayType,
    TableComparisonPlotCollectionType,
    TableComparisonPlotType,
    TableComparisonScoreCollectionType,
    TableComparisonScoreType,
    TabularDataSpecProtocol,
)
from artifact_experiment.base.data_split import DataSplit
from artifact_experiment.table_comparison import TableComparisonPlan
from artifact_experiment.tracking import TrackingClient
from artifact_torch.table_comparison import TableComparisonRoutine

from demos.table_comparison.components.routines.protocols import (
    DemoGenerationParams,
)
from demos.table_comparison.config.constants import (
    ARTIFACT_VALIDATION_PERIOD,
    GENERATION_N_RECORDS,
    GENERATION_TEMPERATURE,
)


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


class DemoTableComparisonRoutine(TableComparisonRoutine[DemoGenerationParams]):
    @classmethod
    def _get_periods(cls) -> Mapping[DataSplit, int]:
        return {DataSplit.TRAIN: ARTIFACT_VALIDATION_PERIOD}

    @classmethod
    def _get_validation_plans(
        cls,
        artifact_resource_spec: TabularDataSpecProtocol,
        tracking_client: Optional[TrackingClient],
    ) -> Mapping[DataSplit, TableComparisonPlan]:
        return {
            DataSplit.TRAIN: DemoTableComparisonPlan.build(
                resource_spec=artifact_resource_spec,
                data_split=DataSplit.TRAIN,
                tracking_client=tracking_client,
            )
        }

    @classmethod
    def _get_generation_params(cls) -> DemoGenerationParams:
        return DemoGenerationParams(
            n_records=GENERATION_N_RECORDS, temperature=GENERATION_TEMPERATURE
        )
