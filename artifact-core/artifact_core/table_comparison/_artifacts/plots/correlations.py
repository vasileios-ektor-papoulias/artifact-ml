from dataclasses import dataclass
from typing import Type, TypeVar, Union

import pandas as pd

from artifact_core._base.core.hyperparams import ArtifactHyperparams
from artifact_core._base.typing.artifact_result import Plot
from artifact_core._libs.artifacts.table_comparison.correlations.calculator import (
    CategoricalAssociationType,
    CategoricalAssociationTypeLiteral,
    ContinuousAssociationType,
    ContinuousAssociationTypeLiteral,
)
from artifact_core._libs.artifacts.table_comparison.correlations.heatmap_plotter import (
    CorrelationHeatmapPlotter,
)
from artifact_core.table_comparison._artifacts.base import (
    TableComparisonPlot,
)
from artifact_core.table_comparison._registries.plots import TableComparisonPlotRegistry
from artifact_core.table_comparison._types.plots import TableComparisonPlotType

CorrelationHeatmapJuxtapositionPlotHyperparamsT = TypeVar(
    "CorrelationHeatmapJuxtapositionPlotHyperparamsT",
    bound="CorrelationHeatmapJuxtapositionPlotHyperparams",
)


@TableComparisonPlotRegistry.register_artifact_hyperparams(
    TableComparisonPlotType.CORRELATION_HEATMAP_JUXTAPOSITION
)
@dataclass(frozen=True)
class CorrelationHeatmapJuxtapositionPlotHyperparams(ArtifactHyperparams):
    categorical_association_type: CategoricalAssociationType
    continuous_association_type: ContinuousAssociationType

    @classmethod
    def build(
        cls: Type[CorrelationHeatmapJuxtapositionPlotHyperparamsT],
        categorical_association_type: Union[
            CategoricalAssociationType, CategoricalAssociationTypeLiteral
        ],
        continuous_association_type: Union[
            ContinuousAssociationType, ContinuousAssociationTypeLiteral
        ],
    ) -> CorrelationHeatmapJuxtapositionPlotHyperparamsT:
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
    TableComparisonPlotType.CORRELATION_HEATMAP_JUXTAPOSITION
)
class CorrelationHeatmapJuxtapositionPlot(
    TableComparisonPlot[CorrelationHeatmapJuxtapositionPlotHyperparams]
):
    def _compare_datasets(
        self, dataset_real: pd.DataFrame, dataset_synthetic: pd.DataFrame
    ) -> Plot:
        plot = CorrelationHeatmapPlotter.get_combined_correlation_heatmaps(
            categorical_correlation_type=self._hyperparams.categorical_association_type,
            continuous_correlation_type=self._hyperparams.continuous_association_type,
            dataset_real=dataset_real,
            dataset_synthetic=dataset_synthetic,
            cat_features=self._resource_spec.cat_features,
        )
        return plot
