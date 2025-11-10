from abc import abstractmethod
from typing import List, Optional, Type

import pandas as pd
from artifact_core.table_comparison.artifacts.base import (
    TableComparisonArtifactResources,
    TabularDataSpecProtocol,
)

from artifact_experiment.base.components.factories.artifact import ArtifactCallbackFactory
from artifact_experiment.base.components.plans.artifact import ArtifactPlan
from artifact_experiment.base.entities.data_split import DataSplit
from artifact_experiment.table_comparison.callback_factory import (
    TableComparisonArrayCollectionType,
    TableComparisonArrayType,
    TableComparisonCallbackFactory,
    TableComparisonPlotCollectionType,
    TableComparisonPlotType,
    TableComparisonScoreCollectionType,
    TableComparisonScoreType,
)


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
