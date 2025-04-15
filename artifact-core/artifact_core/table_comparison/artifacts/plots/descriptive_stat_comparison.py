import pandas as pd
from matplotlib.figure import Figure

from artifact_core.base.artifact_dependencies import NoArtifactHyperparams
from artifact_core.libs.implementation.descriptive_statistics.calculator import (
    DescriptiveStatistic,
)
from artifact_core.libs.implementation.descriptive_statistics.comparison_plots import (
    DescriptiveStatComparisonPlotter,
)
from artifact_core.table_comparison.artifacts.base import (
    TableComparisonPlot,
)
from artifact_core.table_comparison.registries.plots.registry import (
    TableComparisonPlotRegistry,
    TableComparisonPlotType,
)


@TableComparisonPlotRegistry.register_artifact(
    TableComparisonPlotType.DESCRIPTIVE_STATS_COMPARISON_PLOT
)
class ContinuousFeatureDescriptiveStatsComparisonPlot(TableComparisonPlot[NoArtifactHyperparams]):
    def _compare_datasets(
        self, dataset_real: pd.DataFrame, dataset_synthetic: pd.DataFrame
    ) -> Figure:
        plot = DescriptiveStatComparisonPlotter.get_combined_stat_comparison_plot(
            dataset_real=dataset_real,
            dataset_synthetic=dataset_synthetic,
            ls_cts_features=self._data_spec.ls_cts_features,
        )
        return plot


@TableComparisonPlotRegistry.register_artifact(TableComparisonPlotType.MEAN_COMPARISON_PLOT)
class ContinuousFeatureMeanComparisonPlot(TableComparisonPlot[NoArtifactHyperparams]):
    def _compare_datasets(
        self, dataset_real: pd.DataFrame, dataset_synthetic: pd.DataFrame
    ) -> Figure:
        plot = DescriptiveStatComparisonPlotter.get_stat_comparison_plot(
            dataset_real=dataset_real,
            dataset_synthetic=dataset_synthetic,
            ls_cts_features=self._data_spec.ls_cts_features,
            stat=DescriptiveStatistic.MEAN,
        )
        return plot


@TableComparisonPlotRegistry.register_artifact(TableComparisonPlotType.STD_COMPARISON_PLOT)
class ContinuousFeatureSTDComparisonPlot(TableComparisonPlot[NoArtifactHyperparams]):
    def _compare_datasets(
        self, dataset_real: pd.DataFrame, dataset_synthetic: pd.DataFrame
    ) -> Figure:
        plot = DescriptiveStatComparisonPlotter.get_stat_comparison_plot(
            dataset_real=dataset_real,
            dataset_synthetic=dataset_synthetic,
            ls_cts_features=self._data_spec.ls_cts_features,
            stat=DescriptiveStatistic.STD,
        )
        return plot


@TableComparisonPlotRegistry.register_artifact(TableComparisonPlotType.VARIANCE_COMPARISON_PLOT)
class ContinuousFeatureVarianceComparisonPlot(TableComparisonPlot[NoArtifactHyperparams]):
    def _compare_datasets(
        self, dataset_real: pd.DataFrame, dataset_synthetic: pd.DataFrame
    ) -> Figure:
        plot = DescriptiveStatComparisonPlotter.get_stat_comparison_plot(
            dataset_real=dataset_real,
            dataset_synthetic=dataset_synthetic,
            ls_cts_features=self._data_spec.ls_cts_features,
            stat=DescriptiveStatistic.VARIANCE,
        )
        return plot


@TableComparisonPlotRegistry.register_artifact(TableComparisonPlotType.MEDIAN_COMPARISON_PLOT)
class ContinuousFeatureMedianComparisonPlot(TableComparisonPlot[NoArtifactHyperparams]):
    def _compare_datasets(
        self, dataset_real: pd.DataFrame, dataset_synthetic: pd.DataFrame
    ) -> Figure:
        plot = DescriptiveStatComparisonPlotter.get_stat_comparison_plot(
            dataset_real=dataset_real,
            dataset_synthetic=dataset_synthetic,
            ls_cts_features=self._data_spec.ls_cts_features,
            stat=DescriptiveStatistic.MEDIAN,
        )
        return plot


@TableComparisonPlotRegistry.register_artifact(
    TableComparisonPlotType.FIRST_QUARTILE_COMPARISON_PLOT
)
class ContinuousFeatureFirstQuartileComparisonPlot(TableComparisonPlot[NoArtifactHyperparams]):
    def _compare_datasets(
        self, dataset_real: pd.DataFrame, dataset_synthetic: pd.DataFrame
    ) -> Figure:
        plot = DescriptiveStatComparisonPlotter.get_stat_comparison_plot(
            dataset_real=dataset_real,
            dataset_synthetic=dataset_synthetic,
            ls_cts_features=self._data_spec.ls_cts_features,
            stat=DescriptiveStatistic.Q1,
        )
        return plot


@TableComparisonPlotRegistry.register_artifact(
    TableComparisonPlotType.THIRD_QUARTILE_COMPARISON_PLOT
)
class ContinuousFeatureThirdQuartileComparisonPlot(TableComparisonPlot[NoArtifactHyperparams]):
    def _compare_datasets(
        self, dataset_real: pd.DataFrame, dataset_synthetic: pd.DataFrame
    ) -> Figure:
        plot = DescriptiveStatComparisonPlotter.get_stat_comparison_plot(
            dataset_real=dataset_real,
            dataset_synthetic=dataset_synthetic,
            ls_cts_features=self._data_spec.ls_cts_features,
            stat=DescriptiveStatistic.Q3,
        )
        return plot


@TableComparisonPlotRegistry.register_artifact(TableComparisonPlotType.MAX_COMPARISON_PLOT)
class ContinuousFeatureMaximaComparisonPlot(TableComparisonPlot[NoArtifactHyperparams]):
    def _compare_datasets(
        self, dataset_real: pd.DataFrame, dataset_synthetic: pd.DataFrame
    ) -> Figure:
        plot = DescriptiveStatComparisonPlotter.get_stat_comparison_plot(
            dataset_real=dataset_real,
            dataset_synthetic=dataset_synthetic,
            ls_cts_features=self._data_spec.ls_cts_features,
            stat=DescriptiveStatistic.MAX,
        )
        return plot


@TableComparisonPlotRegistry.register_artifact(TableComparisonPlotType.MIN_COMPARISON_PLOT)
class ContinuousFeatureMinimaComparisonPlot(TableComparisonPlot[NoArtifactHyperparams]):
    def _compare_datasets(
        self, dataset_real: pd.DataFrame, dataset_synthetic: pd.DataFrame
    ) -> Figure:
        plot = DescriptiveStatComparisonPlotter.get_stat_comparison_plot(
            dataset_real=dataset_real,
            dataset_synthetic=dataset_synthetic,
            ls_cts_features=self._data_spec.ls_cts_features,
            stat=DescriptiveStatistic.MIN,
        )
        return plot
