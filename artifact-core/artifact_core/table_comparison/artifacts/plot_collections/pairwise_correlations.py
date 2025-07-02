from dataclasses import dataclass
from typing import Dict, Literal, Type, TypeVar, Union

import pandas as pd
from matplotlib.figure import Figure

from artifact_core.base.artifact_dependencies import ArtifactHyperparams
from artifact_core.libs.implementation.tabular.pairwise_correlation.calculator import (
    CategoricalAssociationType,
    ContinuousAssociationType,
)
from artifact_core.libs.implementation.tabular.pairwise_correlation.plotter import (
    PairwiseCorrelationHeatmapPlotter,
)
from artifact_core.libs.resource_spec.tabular.protocol import TabularDataSpecProtocol
from artifact_core.table_comparison.artifacts.base import TableComparisonPlotCollection
from artifact_core.table_comparison.registries.plot_collections.registry import (
    TableComparisonPlotCollectionRegistry,
    TableComparisonPlotCollectionType,
)

correlationPlotCollectionConfigT = TypeVar(
    "correlationPlotCollectionConfigT", bound="CorrelationPlotCollectionConfig"
)


@TableComparisonPlotCollectionRegistry.register_artifact_config(
    TableComparisonPlotCollectionType.PAIRWISE_CORRELATION_PLOTS
)
@dataclass(frozen=True)
class CorrelationPlotCollectionConfig(ArtifactHyperparams):
    categorical_association_type: CategoricalAssociationType
    continuous_association_type: ContinuousAssociationType

    @classmethod
    def build(
        cls: Type[correlationPlotCollectionConfigT],
        categorical_association_type: Union[
            CategoricalAssociationType, Literal["THEILS_U"], Literal["CRAMERS_V"]
        ],
        continuous_association_type: Union[
            ContinuousAssociationType, Literal["PEARSON"], Literal["SPEARMAN"], Literal["KENDALL"]
        ],
    ) -> correlationPlotCollectionConfigT:
        if isinstance(categorical_association_type, str):
            categorical_association_type = CategoricalAssociationType[categorical_association_type]
        if isinstance(continuous_association_type, str):
            continuous_association_type = ContinuousAssociationType[continuous_association_type]
        correlation_plot_collection_hyperparams = cls(
            categorical_association_type=categorical_association_type,
            continuous_association_type=continuous_association_type,
        )
        return correlation_plot_collection_hyperparams


@TableComparisonPlotCollectionRegistry.register_artifact(
    TableComparisonPlotCollectionType.PAIRWISE_CORRELATION_PLOTS
)
class CorrelationPlotCollection(TableComparisonPlotCollection[CorrelationPlotCollectionConfig]):
    def __init__(
        self,
        resource_spec: TabularDataSpecProtocol,
        hyperparams: CorrelationPlotCollectionConfig,
    ):
        self._resource_spec = resource_spec
        self._hyperparams = hyperparams

    def _compare_datasets(
        self, dataset_real: pd.DataFrame, dataset_synthetic: pd.DataFrame
    ) -> Dict[str, Figure]:
        dict_plots = PairwiseCorrelationHeatmapPlotter.get_correlation_plot_collection(
            categorical_correlation_type=self._hyperparams.categorical_association_type,
            continuous_correlation_type=self._hyperparams.continuous_association_type,
            dataset_real=dataset_real,
            dataset_synthetic=dataset_synthetic,
            ls_cat_features=self._resource_spec.ls_cat_features,
        )
        return dict_plots
