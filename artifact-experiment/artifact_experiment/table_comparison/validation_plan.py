from abc import abstractmethod
from typing import List, Type

from artifact_core.table_comparison.artifacts.base import (
    TableComparisonArtifactResources,
    TabularDataSpecProtocol,
)

from artifact_experiment.base.callbacks.factory import ArtifactCallbackFactory
from artifact_experiment.base.validation_plan import ArtifactValidationPlan
from artifact_experiment.table_comparison.callback_factory import (
    TableComparisonArrayCollectionType,
    TableComparisonArrayType,
    TableComparisonCallbackFactory,
    TableComparisonPlotCollectionType,
    TableComparisonPlotType,
    TableComparisonScoreCollectionType,
    TableComparisonScoreType,
)


class TableComparisonValidationPlan(
    ArtifactValidationPlan[
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
