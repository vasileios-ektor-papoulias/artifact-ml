from dataclasses import dataclass
from typing import Literal, Type, TypeVar, Union

import pandas as pd
from matplotlib.figure import Figure

from artifact_core.base.artifact_dependencies import ArtifactHyperparams
from artifact_core.libs.implementation.tabular.correlations.calculator import (
    CategoricalAssociationType,
    ContinuousAssociationType,
)
from artifact_core.libs.implementation.tabular.correlations.heatmap_plotter import (
    CorrelationHeatmapPlotter,
)
from artifact_core.table_comparison.artifacts.base import (
    TableComparisonPlot,
)
from artifact_core.table_comparison.registries.plots.registry import (
    TableComparisonPlotRegistry,
    TableComparisonPlotType,
)

CorrelationHeatmapsHyperparamsT = TypeVar(
    "CorrelationHeatmapsHyperparamsT", bound="CorrelationHeatmapsHyperparams"
)


@TableComparisonPlotRegistry.register_artifact_config(TableComparisonPlotType.CORRELATION_HEATMAPS)
@dataclass(frozen=True)
class CorrelationHeatmapsHyperparams(ArtifactHyperparams):
    categorical_association_type: CategoricalAssociationType
    continuous_association_type: ContinuousAssociationType

    @classmethod
    def build(
        cls: Type[CorrelationHeatmapsHyperparamsT],
        categorical_association_type: Union[
            CategoricalAssociationType, Literal["THEILS_U"], Literal["CRAMERS_V"]
        ],
        continuous_association_type: Union[
            ContinuousAssociationType, Literal["PEARSON"], Literal["SPEARMAN"], Literal["KENDALL"]
        ],
    ) -> CorrelationHeatmapsHyperparamsT:
        if isinstance(categorical_association_type, str):
            categorical_association_type = CategoricalAssociationType[categorical_association_type]
        if isinstance(continuous_association_type, str):
            continuous_association_type = ContinuousAssociationType[continuous_association_type]
        correlation_comparison_heatmap_hyperparams = cls(
            categorical_association_type=categorical_association_type,
            continuous_association_type=continuous_association_type,
        )
        return correlation_comparison_heatmap_hyperparams


@TableComparisonPlotRegistry.register_artifact(TableComparisonPlotType.CORRELATION_HEATMAPS)
class CorrelationHeatmaps(TableComparisonPlot[CorrelationHeatmapsHyperparams]):
    def _compare_datasets(
        self, dataset_real: pd.DataFrame, dataset_synthetic: pd.DataFrame
    ) -> Figure:
        plot = CorrelationHeatmapPlotter.get_combined_correlation_heatmaps(
            categorical_correlation_type=self._hyperparams.categorical_association_type,
            continuous_correlation_type=self._hyperparams.continuous_association_type,
            dataset_real=dataset_real,
            dataset_synthetic=dataset_synthetic,
            ls_cat_features=self._resource_spec.ls_cat_features,
        )
        return plot
