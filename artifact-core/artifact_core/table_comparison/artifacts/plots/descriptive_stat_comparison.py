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
