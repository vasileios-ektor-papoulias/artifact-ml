from dataclasses import dataclass
from typing import Dict, Type, TypeVar, Union

import pandas as pd
from matplotlib.figure import Figure

from artifact_core.base.artifact_dependencies import ArtifactHyperparams
from artifact_core.libs.implementation.tabular.correlations.calculator import (
    CategoricalAssociationType,
    CategoricalAssociationTypeLiteral,
    ContinuousAssociationType,
    ContinuousAssociationTypeLiteral,
)
from artifact_core.libs.implementation.tabular.correlations.heatmap_plotter import (
    CorrelationHeatmapPlotter,
)
from artifact_core.table_comparison.artifacts.base import TableComparisonPlotCollection
from artifact_core.table_comparison.registries.plot_collections.registry import (
    TableComparisonPlotCollectionRegistry,
    TableComparisonPlotCollectionType,
)

CorrelationHeatmapsHyperparamsT = TypeVar(
    "CorrelationHeatmapsHyperparamsT", bound="CorrelationHeatmapsHyperparams"
)


@TableComparisonPlotCollectionRegistry.register_artifact_hyperparams(
    TableComparisonPlotCollectionType.CORRELATION_HEATMAPS
)
@dataclass(frozen=True)
class CorrelationHeatmapsHyperparams(ArtifactHyperparams):
    categorical_association_type: CategoricalAssociationType
    continuous_association_type: ContinuousAssociationType

    @classmethod
    def build(
        cls: Type[CorrelationHeatmapsHyperparamsT],
        categorical_association_type: Union[
            CategoricalAssociationType, CategoricalAssociationTypeLiteral
        ],
        continuous_association_type: Union[
            ContinuousAssociationType, ContinuousAssociationTypeLiteral
        ],
    ) -> CorrelationHeatmapsHyperparamsT:
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
    TableComparisonPlotCollectionType.CORRELATION_HEATMAPS
)
class CorrelationHeatmaps(TableComparisonPlotCollection[CorrelationHeatmapsHyperparams]):
    def _compare_datasets(
        self, dataset_real: pd.DataFrame, dataset_synthetic: pd.DataFrame
    ) -> Dict[str, Figure]:
        dict_plots = CorrelationHeatmapPlotter.get_correlation_heatmap_collection(
            categorical_correlation_type=self._hyperparams.categorical_association_type,
            continuous_correlation_type=self._hyperparams.continuous_association_type,
            dataset_real=dataset_real,
            dataset_synthetic=dataset_synthetic,
            ls_cat_features=self._resource_spec.ls_cat_features,
        )
        return dict_plots
