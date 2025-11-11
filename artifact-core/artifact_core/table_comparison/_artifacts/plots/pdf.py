import pandas as pd
from matplotlib.figure import Figure

from artifact_core._base.artifact_dependencies import NoArtifactHyperparams
from artifact_core._libs.implementation.tabular.pdf.overlaid_plotter import (
    TabularOverlaidPDFPlotter,
)
from artifact_core.table_comparison._artifacts.base import (
    TableComparisonPlot,
)
from artifact_core.table_comparison._registries.plots.registry import (
    TableComparisonPlotRegistry,
    TableComparisonPlotType,
)


@TableComparisonPlotRegistry.register_artifact(TableComparisonPlotType.PDF)
class PDFPlot(TableComparisonPlot[NoArtifactHyperparams]):
    def _compare_datasets(
        self, dataset_real: pd.DataFrame, dataset_synthetic: pd.DataFrame
    ) -> Figure:
        dict_plots = TabularOverlaidPDFPlotter.get_overlaid_pdf_plot(
            dataset_real=dataset_real,
            dataset_synthetic=dataset_synthetic,
            ls_features_order=self._resource_spec.ls_features,
            ls_cts_features=self._resource_spec.ls_cts_features,
            ls_cat_features=self._resource_spec.ls_cat_features,
            cat_unique_map=self._resource_spec.cat_unique_map,
        )
        return dict_plots
