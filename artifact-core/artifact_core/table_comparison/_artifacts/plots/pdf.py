import pandas as pd

from artifact_core._base.core.hyperparams import NoArtifactHyperparams
from artifact_core._base.typing.artifact_result import Plot
from artifact_core._libs.artifacts.table_comparison.pdf.overlaid_plotter import (
    TabularOverlaidPDFPlotter,
)
from artifact_core.table_comparison._artifacts.base import TableComparisonPlot
from artifact_core.table_comparison._registries.plots import TableComparisonPlotRegistry
from artifact_core.table_comparison._types.plots import TableComparisonPlotType


@TableComparisonPlotRegistry.register_artifact(TableComparisonPlotType.PDF)
class PDFPlot(TableComparisonPlot[NoArtifactHyperparams]):
    def _compare_datasets(
        self, dataset_real: pd.DataFrame, dataset_synthetic: pd.DataFrame
    ) -> Plot:
        dict_plots = TabularOverlaidPDFPlotter.get_overlaid_pdf_plot(
            dataset_real=dataset_real,
            dataset_synthetic=dataset_synthetic,
            features_order=self._resource_spec.features,
            cts_features=self._resource_spec.cts_features,
            cat_features=self._resource_spec.cat_features,
            cat_unique_map=self._resource_spec.cat_unique_map,
        )
        return dict_plots
