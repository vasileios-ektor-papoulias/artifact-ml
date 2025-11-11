from dataclasses import dataclass

import pandas as pd
from matplotlib.figure import Figure

from artifact_core._base.artifact_dependencies import ArtifactHyperparams
from artifact_core._libs.implementation.tabular.projections.truncated_svd import (
    TruncatedSVDHyperparams,
    TruncatedSVDProjector,
)
from artifact_core.table_comparison._artifacts.base import (
    TableComparisonPlot,
)
from artifact_core.table_comparison._registries.plots.registry import (
    TableComparisonPlotRegistry,
    TableComparisonPlotType,
)


@TableComparisonPlotRegistry.register_artifact_hyperparams(
    TableComparisonPlotType.TRUNCATED_SVD_JUXTAPOSITION
)
@dataclass(frozen=True)
class TruncatedSVDJuxtapositionPlotHyperparams(ArtifactHyperparams):
    use_categorical: bool


@TableComparisonPlotRegistry.register_artifact(TableComparisonPlotType.TRUNCATED_SVD_JUXTAPOSITION)
class TruncatedSVDJuxtapositionPlot(TableComparisonPlot[TruncatedSVDJuxtapositionPlotHyperparams]):
    def _compare_datasets(
        self, dataset_real: pd.DataFrame, dataset_synthetic: pd.DataFrame
    ) -> Figure:
        projector_config = TruncatedSVDHyperparams(
            use_categorical=self._hyperparams.use_categorical
        )
        projector = TruncatedSVDProjector.build(
            ls_cat_features=self._resource_spec.ls_cat_features,
            ls_cts_features=self._resource_spec.ls_cts_features,
            projector_config=projector_config,
        )
        plot = projector.produce_projection_comparison_plot(
            dataset_real=dataset_real, dataset_synthetic=dataset_synthetic
        )
        return plot
