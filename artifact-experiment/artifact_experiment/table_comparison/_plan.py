from abc import abstractmethod
from typing import List, Optional, Type

import pandas as pd
from artifact_core.table_comparison import (
    TableComparisonArrayCollectionType,
    TableComparisonArrayType,
    TableComparisonPlotCollectionType,
    TableComparisonPlotType,
    TableComparisonScoreCollectionType,
    TableComparisonScoreType,
)
from artifact_core.table_comparison.spi import (
    TableComparisonArtifactResources,
    TabularDataSpecProtocol,
)

from artifact_experiment._base.components.callbacks.export import ExportCallback
from artifact_experiment._base.components.factories.artifact import ArtifactCallbackFactory
from artifact_experiment._base.components.plans.artifact import (
    ArtifactPlan,
    ArtifactPlanBuildContext,
)
from artifact_experiment._base.components.resources.artifact import ArtifactCallbackResources
from artifact_experiment._base.components.resources.export import ExportCallbackResources
from artifact_experiment._base.primitives.data_split import DataSplit
from artifact_experiment._impl.callbacks.table_export import TableExportCallback
from artifact_experiment.table_comparison._callback_factory import TableComparisonCallbackFactory


class TableComparisonPlan(
    ArtifactPlan[
        TableComparisonArtifactResources,
        TabularDataSpecProtocol,
        TableComparisonScoreType,
        TableComparisonArrayType,
        TableComparisonPlotType,
        TableComparisonScoreCollectionType,
        TableComparisonArrayCollectionType,
        TableComparisonPlotCollectionType,
        pd.DataFrame,
    ]
):
    @staticmethod
    @abstractmethod
    def _get_score_types() -> List[TableComparisonScoreType]: ...

    @staticmethod
    @abstractmethod
    def _get_array_types() -> List[TableComparisonArrayType]: ...

    @staticmethod
    @abstractmethod
    def _get_plot_types() -> List[TableComparisonPlotType]: ...

    @staticmethod
    @abstractmethod
    def _get_score_collection_types() -> List[TableComparisonScoreCollectionType]: ...

    @staticmethod
    @abstractmethod
    def _get_array_collection_types() -> List[TableComparisonArrayCollectionType]: ...

    @staticmethod
    @abstractmethod
    def _get_plot_collection_types() -> List[TableComparisonPlotCollectionType]: ...

    def execute_table_comparison(
        self,
        dataset_real: pd.DataFrame,
        dataset_synthetic: pd.DataFrame,
        data_split: Optional[DataSplit] = None,
    ):
        artifact_resources = TableComparisonArtifactResources(
            dataset_real=dataset_real, dataset_synthetic=dataset_synthetic
        )
        super().execute_artifacts(resources=artifact_resources, data_split=data_split)

    @staticmethod
    def _get_callback_factory() -> Type[
        ArtifactCallbackFactory[
            TableComparisonArtifactResources,
            TabularDataSpecProtocol,
            TableComparisonScoreType,
            TableComparisonArrayType,
            TableComparisonPlotType,
            TableComparisonScoreCollectionType,
            TableComparisonArrayCollectionType,
            TableComparisonPlotCollectionType,
        ]
    ]:
        return TableComparisonCallbackFactory

    @classmethod
    def _get_export_callback(
        cls, context: ArtifactPlanBuildContext[TabularDataSpecProtocol]
    ) -> Optional[ExportCallback[ExportCallbackResources[pd.DataFrame]]]:
        if context.file_writer is not None:
            return TableExportCallback(writer=context.file_writer)

    @classmethod
    def _get_export_resources(
        cls, resources: ArtifactCallbackResources[TableComparisonArtifactResources]
    ) -> ExportCallbackResources[pd.DataFrame]:
        export_resources = ExportCallbackResources(
            export_data=resources.artifact_resources.dataset_synthetic, trigger="SYNTHETIC"
        )
        return export_resources
