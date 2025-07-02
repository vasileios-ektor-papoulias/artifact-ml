from dataclasses import dataclass

import pandas as pd
from matplotlib.figure import Figure

from artifact_core.base.artifact_dependencies import ArtifactHyperparams
from artifact_core.libs.implementation.tabular.projections.pca import (
    PCAHyperparams,
    PCAProjector,
)
from artifact_core.table_comparison.artifacts.base import (
    TableComparisonPlot,
)
from artifact_core.table_comparison.registries.plots.registry import (
    TableComparisonPlotRegistry,
    TableComparisonPlotType,
)


@TableComparisonPlotRegistry.register_artifact_config(TableComparisonPlotType.PCA_PROJECTION_PLOT)
@dataclass(frozen=True)
class PCAProjectionComparisonPlotConfig(ArtifactHyperparams):
    use_categorical: bool


@TableComparisonPlotRegistry.register_artifact(TableComparisonPlotType.PCA_PROJECTION_PLOT)
class PCAProjectionComparisonPlot(TableComparisonPlot[PCAProjectionComparisonPlotConfig]):
    def _compare_datasets(
        self, dataset_real: pd.DataFrame, dataset_synthetic: pd.DataFrame
    ) -> Figure:
        projector_config = PCAHyperparams(use_categorical=self._hyperparams.use_categorical)
        projector = PCAProjector.build(
            ls_cat_features=self._resource_spec.ls_cat_features,
            ls_cts_features=self._resource_spec.ls_cts_features,
            projector_config=projector_config,
        )
        plot = projector.produce_projection_comparison_plot(
            dataset_real=dataset_real, dataset_synthetic=dataset_synthetic
        )
        return plot
