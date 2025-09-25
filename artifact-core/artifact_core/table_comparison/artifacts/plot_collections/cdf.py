from typing import Dict

import pandas as pd
from matplotlib.figure import Figure

from artifact_core.base.artifact_dependencies import NoArtifactHyperparams
from artifact_core.libs.implementation.tabular.cdf.overlaid_plotter import TabularOverlaidCDFPlotter
from artifact_core.table_comparison.artifacts.base import (
    TableComparisonPlotCollection,
)
from artifact_core.table_comparison.registries.plot_collections.registry import (
    TableComparisonPlotCollectionRegistry,
    TableComparisonPlotCollectionType,
)


@TableComparisonPlotCollectionRegistry.register_artifact(TableComparisonPlotCollectionType.CDF)
class CDFPlots(TableComparisonPlotCollection[NoArtifactHyperparams]):
    def _compare_datasets(
        self, dataset_real: pd.DataFrame, dataset_synthetic: pd.DataFrame
    ) -> Dict[str, Figure]:
        dict_plots = TabularOverlaidCDFPlotter.get_overlaid_cdf_plot_collection(
            dataset_real=dataset_real,
            dataset_synthetic=dataset_synthetic,
            ls_cts_features=self._resource_spec.ls_cts_features,
        )
        return dict_plots
