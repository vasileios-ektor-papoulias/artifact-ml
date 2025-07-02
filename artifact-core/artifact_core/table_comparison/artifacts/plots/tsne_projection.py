from dataclasses import dataclass

import pandas as pd
from matplotlib.figure import Figure

from artifact_core.base.artifact_dependencies import ArtifactHyperparams
from artifact_core.libs.implementation.tabular.projections.tsne import (
    TSNEHyperparams,
    TSNEProjector,
)
from artifact_core.table_comparison.artifacts.base import (
    TableComparisonPlot,
)
from artifact_core.table_comparison.registries.plots.registry import (
    TableComparisonPlotRegistry,
    TableComparisonPlotType,
)


@TableComparisonPlotRegistry.register_artifact_config(TableComparisonPlotType.TSNE_PROJECTION_PLOT)
@dataclass(frozen=True)
class TSNEProjectionComparisonPlotConfig(ArtifactHyperparams):
    use_categorical: bool
    perplexity: float
    learning_rate: float | str
    max_iter: int


@TableComparisonPlotRegistry.register_artifact(TableComparisonPlotType.TSNE_PROJECTION_PLOT)
class TSNEProjectionComparisonPlot(TableComparisonPlot[TSNEProjectionComparisonPlotConfig]):
    def _compare_datasets(
        self, dataset_real: pd.DataFrame, dataset_synthetic: pd.DataFrame
    ) -> Figure:
        projector_config = TSNEHyperparams(use_categorical=self._hyperparams.use_categorical)
        projector = TSNEProjector.build(
            ls_cat_features=self._resource_spec.ls_cat_features,
            ls_cts_features=self._resource_spec.ls_cts_features,
            projector_config=projector_config,
        )
        plot = projector.produce_projection_comparison_plot(
            dataset_real=dataset_real, dataset_synthetic=dataset_synthetic
        )
        return plot
