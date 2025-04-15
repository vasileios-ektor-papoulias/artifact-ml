import pandas as pd
from matplotlib.figure import Figure

from artifact_core.base.artifact_dependencies import NoArtifactHyperparams
from artifact_core.libs.implementation.cdf.overlaid_plotter import (
    OverlaidCDFPlotter,
)
from artifact_core.table_comparison.artifacts.base import (
    TableComparisonPlot,
)
class CDFComparisonCombinedPlot(TableComparisonPlot[NoArtifactHyperparams]):
    def _compare_datasets(
        self, dataset_real: pd.DataFrame, dataset_synthetic: pd.DataFrame
    ) -> Figure:
        dict_plots = OverlaidCDFPlotter.get_overlaid_cdf_plot(
            dataset_real=dataset_real,
            dataset_synthetic=dataset_synthetic,
            ls_cts_features=self._data_spec.ls_cts_features,
        )
        return dict_plots
