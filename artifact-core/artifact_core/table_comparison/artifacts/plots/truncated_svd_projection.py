from dataclasses import dataclass

import pandas as pd
from matplotlib.figure import Figure

from artifact_core.base.artifact_dependencies import ArtifactHyperparams
from artifact_core.libs.data_spec.tabular.protocol import (
    TabularDataSpecProtocol,
)
from artifact_core.libs.implementation.projections.truncated_svd import (
    TruncatedSVDHyperparams,
    TruncatedSVDProjector,
)
from artifact_core.table_comparison.artifacts.base import (
    TableComparisonPlot,
)
from artifact_core.table_comparison.registries.plots.registry import (
    TableComparisonPlotRegistry,
    TableComparisonPlotType,
)


@TableComparisonPlotRegistry.register_artifact_config(
    TableComparisonPlotType.TRUNCATED_SVD_PROJECTION_PLOT
)
@dataclass(frozen=True)
class TruncatedSVDProjectionComparisonPlotConfig(ArtifactHyperparams):
    use_categorical: bool


@TableComparisonPlotRegistry.register_artifact(
    TableComparisonPlotType.TRUNCATED_SVD_PROJECTION_PLOT
)
class TruncatedSVDProjectionComparisonPlot(
    TableComparisonPlot[TruncatedSVDProjectionComparisonPlotConfig]
):
    def __init__(
        self,
        data_spec: TabularDataSpecProtocol,
        hyperparams: TruncatedSVDProjectionComparisonPlotConfig,
    ):
        self._data_spec = data_spec
        self._hyperparams = hyperparams

    def _compare_datasets(
        self, dataset_real: pd.DataFrame, dataset_synthetic: pd.DataFrame
    ) -> Figure:
        projector_config = TruncatedSVDHyperparams(
            use_categorical=self._hyperparams.use_categorical
        )
        projector = TruncatedSVDProjector.build(
            ls_cat_features=self._data_spec.ls_cat_features,
            ls_cts_features=self._data_spec.ls_cts_features,
            projector_config=projector_config,
        )
        plot = projector.produce_projection_comparison_plot(
            dataset_real=dataset_real, dataset_synthetic=dataset_synthetic
        )
        return plot
