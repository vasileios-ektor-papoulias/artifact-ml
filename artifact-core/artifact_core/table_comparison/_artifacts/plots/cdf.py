import pandas as pd

from artifact_core._base.core.hyperparams import NoArtifactHyperparams
from artifact_core._base.typing.artifact_result import Plot
from artifact_core._libs.artifacts.table_comparison.cdf.overlaid_plotter import (
    TabularOverlaidCDFPlotter,
)
from artifact_core.table_comparison._artifacts.base import (
    TableComparisonPlot,
)
from artifact_core.table_comparison._registries.plots import TableComparisonPlotRegistry
from artifact_core.table_comparison._types.plots import TableComparisonPlotType


@TableComparisonPlotRegistry.register_artifact(TableComparisonPlotType.CDF)
class CDFPlot(TableComparisonPlot[NoArtifactHyperparams]):
    def _compare_datasets(
        self, dataset_real: pd.DataFrame, dataset_synthetic: pd.DataFrame
    ) -> Plot:
        dict_plots = TabularOverlaidCDFPlotter.get_overlaid_cdf_plot(
            dataset_real=dataset_real,
            dataset_synthetic=dataset_synthetic,
            cts_features=self._resource_spec.cts_features,
        )
        return dict_plots
