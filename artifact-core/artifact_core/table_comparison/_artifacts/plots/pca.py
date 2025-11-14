from dataclasses import dataclass

import pandas as pd

from artifact_core._base.core.hyperparams import ArtifactHyperparams
from artifact_core._base.typing.artifact_result import Plot
from artifact_core._libs.artifacts.table_comparison.projections.pca import (
    PCAHyperparams,
    PCAProjector,
)
from artifact_core.table_comparison._artifacts.base import TableComparisonPlot
from artifact_core.table_comparison._registries.plots import TableComparisonPlotRegistry
from artifact_core.table_comparison._types.plots import TableComparisonPlotType


@TableComparisonPlotRegistry.register_artifact_hyperparams(
    TableComparisonPlotType.PCA_JUXTAPOSITION
)
@dataclass(frozen=True)
class PCAJuxtapositionPlotHyperparams(ArtifactHyperparams):
    use_categorical: bool


@TableComparisonPlotRegistry.register_artifact(TableComparisonPlotType.PCA_JUXTAPOSITION)
class PCAJuxtapositionPlot(TableComparisonPlot[PCAJuxtapositionPlotHyperparams]):
    def _compare_datasets(
        self, dataset_real: pd.DataFrame, dataset_synthetic: pd.DataFrame
    ) -> Plot:
        projector_config = PCAHyperparams(use_categorical=self._hyperparams.use_categorical)
        projector = PCAProjector.build(
            cat_features=self._resource_spec.cat_features,
            cts_features=self._resource_spec.cts_features,
            projector_config=projector_config,
        )
        plot = projector.produce_projection_comparison_plot(
            dataset_real=dataset_real, dataset_synthetic=dataset_synthetic
        )
        return plot
