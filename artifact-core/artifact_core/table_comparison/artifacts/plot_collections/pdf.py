from typing import Dict

import pandas as pd
from matplotlib.figure import Figure

from artifact_core.base.artifact_dependencies import NoArtifactHyperparams
from artifact_core.libs.implementation.tabular.pdf.overlaid_plotter import (
    OverlaidPDFPlotter,
)
from artifact_core.table_comparison.artifacts.base import (
    TableComparisonPlotCollection,
)
from artifact_core.table_comparison.registries.plot_collections.registry import (
    TableComparisonPlotCollectionRegistry,
    TableComparisonPlotCollectionType,
)


@TableComparisonPlotCollectionRegistry.register_artifact(TableComparisonPlotCollectionType.PDF)
class PDFPlots(TableComparisonPlotCollection[NoArtifactHyperparams]):
    def _compare_datasets(
        self, dataset_real: pd.DataFrame, dataset_synthetic: pd.DataFrame
    ) -> Dict[str, Figure]:
        dict_plots = OverlaidPDFPlotter.get_overlaid_pdf_plot_collection(
            dataset_real=dataset_real,
            dataset_synthetic=dataset_synthetic,
            ls_features_order=self._resource_spec.ls_features,
            ls_cts_features=self._resource_spec.ls_cts_features,
            ls_cat_features=self._resource_spec.ls_cat_features,
            cat_unique_map=self._resource_spec.cat_unique_map,
        )
        return dict_plots
