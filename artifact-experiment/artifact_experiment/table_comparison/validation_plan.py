from abc import abstractmethod
from typing import List, Type

import pandas as pd
from artifact_core.table_comparison.artifacts.base import (
    TableComparisonArtifactResources,
    TabularDataSpecProtocol,
)

from artifact_experiment.base.callback_factory import ArtifactCallbackFactory
from artifact_experiment.base.validation_plan import ValidationPlan
from artifact_experiment.table_comparison.callback_factory import (
    TableComparisonArrayCollectionType,
    TableComparisonArrayType,
    TableComparisonCallbackFactory,
    TableComparisonPlotCollectionType,
    TableComparisonPlotType,
    TableComparisonScoreCollectionType,
    TableComparisonScoreType,
)
from artifact_experiment.table_comparison.callback_resources import TableComparisonCallbackResources


class TableComparisonPlan(
    ValidationPlan[
        TableComparisonScoreType,
        TableComparisonArrayType,
        TableComparisonPlotType,
        TableComparisonScoreCollectionType,
        TableComparisonArrayCollectionType,
        TableComparisonPlotCollectionType,
        TableComparisonArtifactResources,
        TabularDataSpecProtocol,
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

    def execute_table_comparison(self, dataset_real: pd.DataFrame, dataset_synthetic: pd.DataFrame):
        callback_resources = TableComparisonCallbackResources.build(
            dataset_real=dataset_real, dataset_synthetic=dataset_synthetic
        )
        super().execute(resources=callback_resources)

    @staticmethod
    def _get_callback_factory() -> Type[
        ArtifactCallbackFactory[
            TableComparisonScoreType,
            TableComparisonArrayType,
            TableComparisonPlotType,
            TableComparisonScoreCollectionType,
            TableComparisonArrayCollectionType,
            TableComparisonPlotCollectionType,
            TableComparisonArtifactResources,
            TabularDataSpecProtocol,
        ]
    ]:
        return TableComparisonCallbackFactory
