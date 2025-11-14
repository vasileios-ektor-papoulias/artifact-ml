from typing import Dict

import pandas as pd
from matplotlib.figure import Figure

from artifact_core._base.core.hyperparams import NoArtifactHyperparams
from artifact_core._libs.artifacts.table_comparison.pdf.overlaid_plotter import (
    TabularOverlaidPDFPlotter,
)
from artifact_core.table_comparison._artifacts.base import TableComparisonPlotCollection
from artifact_core.table_comparison._registries.plot_collections import (
    TableComparisonPlotCollectionRegistry,
)
from artifact_core.table_comparison._types.plot_collections import TableComparisonPlotCollectionType


@TableComparisonPlotCollectionRegistry.register_artifact(TableComparisonPlotCollectionType.PDF)
class PDFPlots(TableComparisonPlotCollection[NoArtifactHyperparams]):
    def _compare_datasets(
        self, dataset_real: pd.DataFrame, dataset_synthetic: pd.DataFrame
    ) -> Dict[str, Figure]:
        dict_plots = TabularOverlaidPDFPlotter.get_overlaid_pdf_plot_collection(
            dataset_real=dataset_real,
            dataset_synthetic=dataset_synthetic,
            features_order=self._resource_spec.features,
            cts_features=self._resource_spec.cts_features,
            cat_features=self._resource_spec.cat_features,
            cat_unique_map=self._resource_spec.cat_unique_map,
        )
        return dict_plots
