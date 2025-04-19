from dataclasses import dataclass
from typing import Literal, Type, TypeVar, Union

import pandas as pd
from matplotlib.figure import Figure

from artifact_core.base.artifact_dependencies import ArtifactHyperparams
from artifact_core.libs.data_spec.tabular.protocol import (
    TabularDataSpecProtocol,
)
from artifact_core.libs.implementation.pairwsie_correlation.calculator import (
    CategoricalAssociationType,
    ContinuousAssociationType,
)
from artifact_core.libs.implementation.pairwsie_correlation.plotter import (
    PairwiseCorrelationHeatmapPlotter,
)
from artifact_core.table_comparison.artifacts.base import (
    TableComparisonPlot,
)
from artifact_core.table_comparison.registries.plots.registry import (
    TableComparisonPlotRegistry,
    TableComparisonPlotType,
)

correlationComparisonHeatmapConfigT = TypeVar(
    "correlationComparisonHeatmapConfigT", bound="CorrelationComparisonHeatmapConfig"
)


@TableComparisonPlotRegistry.register_artifact_config(
    TableComparisonPlotType.PAIRWISE_CORRELATION_COMPARISON_HEATMAP
)
@dataclass(frozen=True)
class CorrelationComparisonHeatmapConfig(ArtifactHyperparams):
    categorical_association_type: CategoricalAssociationType
    continuous_association_type: ContinuousAssociationType

    @classmethod
    def build(
        cls: Type[correlationComparisonHeatmapConfigT],
        categorical_association_type: Union[
            CategoricalAssociationType, Literal["THEILS_U"], Literal["CRAMERS_V"]
        ],
        continuous_association_type: Union[
            ContinuousAssociationType, Literal["PEARSON"], Literal["SPEARMAN"], Literal["KENDALL"]
        ],
    ) -> correlationComparisonHeatmapConfigT:
        if isinstance(categorical_association_type, str):
            categorical_association_type = CategoricalAssociationType[categorical_association_type]
        if isinstance(continuous_association_type, str):
            continuous_association_type = ContinuousAssociationType[continuous_association_type]
        correlation_comparison_heatmap_hyperparams = cls(
            categorical_association_type=categorical_association_type,
            continuous_association_type=continuous_association_type,
        )
        return correlation_comparison_heatmap_hyperparams


@TableComparisonPlotRegistry.register_artifact(
    TableComparisonPlotType.PAIRWISE_CORRELATION_COMPARISON_HEATMAP
)
class CorrelationComparisonCombinedPlot(TableComparisonPlot[CorrelationComparisonHeatmapConfig]):
    def __init__(
        self,
        data_spec: TabularDataSpecProtocol,
        hyperparams: CorrelationComparisonHeatmapConfig,
    ):
        self._data_spec = data_spec
        self._hyperparams = hyperparams

    def _compare_datasets(
        self, dataset_real: pd.DataFrame, dataset_synthetic: pd.DataFrame
    ) -> Figure:
        plot = PairwiseCorrelationHeatmapPlotter.get_combined_correlation_plot(
            categorical_correlation_type=self._hyperparams.categorical_association_type,
            continuous_correlation_type=self._hyperparams.continuous_association_type,
            dataset_real=dataset_real,
            dataset_synthetic=dataset_synthetic,
            ls_cat_features=self._data_spec.ls_cat_features,
        )
        return plot
