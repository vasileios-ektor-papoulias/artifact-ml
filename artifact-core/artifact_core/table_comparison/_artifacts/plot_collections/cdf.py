import pandas as pd

from artifact_core._base.core.hyperparams import NoArtifactHyperparams
from artifact_core._base.typing.artifact_result import PlotCollection
from artifact_core._libs.artifacts.table_comparison.cdf.overlaid_plotter import (
    TabularOverlaidCDFPlotter,
)
from artifact_core.table_comparison._artifacts.base import TableComparisonPlotCollection
from artifact_core.table_comparison._registries.plot_collections import (
    TableComparisonPlotCollectionRegistry,
)
from artifact_core.table_comparison._types.plot_collections import TableComparisonPlotCollectionType


@TableComparisonPlotCollectionRegistry.register_artifact(TableComparisonPlotCollectionType.CDF)
class CDFPlots(TableComparisonPlotCollection[NoArtifactHyperparams]):
    def _compare_datasets(
        self, dataset_real: pd.DataFrame, dataset_synthetic: pd.DataFrame
    ) -> PlotCollection:
        dict_plots = TabularOverlaidCDFPlotter.get_overlaid_cdf_plot_collection(
            dataset_real=dataset_real,
            dataset_synthetic=dataset_synthetic,
            cts_features=self._resource_spec.cts_features,
        )
        return dict_plots
