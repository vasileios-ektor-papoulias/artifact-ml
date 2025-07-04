import pandas as pd
from matplotlib.figure import Figure

from artifact_core.base.artifact_dependencies import NoArtifactHyperparams
from artifact_core.libs.implementation.tabular.descriptive_stats.alignment_plotter import (
    DescriptiveStatsAlignmentPlotter,
)
from artifact_core.libs.implementation.tabular.descriptive_stats.calculator import (
    DescriptiveStatistic,
)
from artifact_core.table_comparison.artifacts.base import (
    TableComparisonPlot,
)
from artifact_core.table_comparison.registries.plots.registry import (
    TableComparisonPlotRegistry,
    TableComparisonPlotType,
)


@TableComparisonPlotRegistry.register_artifact(
    TableComparisonPlotType.DESCRIPTIVE_STATS_ALIGNMENT_PLOT
)
class DescriptiveStatsAlignmentPlot(TableComparisonPlot[NoArtifactHyperparams]):
    def _compare_datasets(
        self, dataset_real: pd.DataFrame, dataset_synthetic: pd.DataFrame
    ) -> Figure:
        plot = DescriptiveStatsAlignmentPlotter.get_combined_stat_alignment_plot(
            dataset_real=dataset_real,
            dataset_synthetic=dataset_synthetic,
            ls_cts_features=self._resource_spec.ls_cts_features,
        )
        return plot


@TableComparisonPlotRegistry.register_artifact(TableComparisonPlotType.MEAN_ALIGNMENT_PLOT)
class MeanAlignmentPlot(TableComparisonPlot[NoArtifactHyperparams]):
    def _compare_datasets(
        self, dataset_real: pd.DataFrame, dataset_synthetic: pd.DataFrame
    ) -> Figure:
        plot = DescriptiveStatsAlignmentPlotter.get_single_stat_alignment_plot(
            dataset_real=dataset_real,
            dataset_synthetic=dataset_synthetic,
            ls_cts_features=self._resource_spec.ls_cts_features,
            stat=DescriptiveStatistic.MEAN,
        )
        return plot


@TableComparisonPlotRegistry.register_artifact(TableComparisonPlotType.STD_ALIGNMENT_PLOT)
class STDAlignmentPlot(TableComparisonPlot[NoArtifactHyperparams]):
    def _compare_datasets(
        self, dataset_real: pd.DataFrame, dataset_synthetic: pd.DataFrame
    ) -> Figure:
        plot = DescriptiveStatsAlignmentPlotter.get_single_stat_alignment_plot(
            dataset_real=dataset_real,
            dataset_synthetic=dataset_synthetic,
            ls_cts_features=self._resource_spec.ls_cts_features,
            stat=DescriptiveStatistic.STD,
        )
        return plot


@TableComparisonPlotRegistry.register_artifact(TableComparisonPlotType.VARIANCE_ALIGNMENT_PLOT)
class VarianceAlignmentPlot(TableComparisonPlot[NoArtifactHyperparams]):
    def _compare_datasets(
        self, dataset_real: pd.DataFrame, dataset_synthetic: pd.DataFrame
    ) -> Figure:
        plot = DescriptiveStatsAlignmentPlotter.get_single_stat_alignment_plot(
            dataset_real=dataset_real,
            dataset_synthetic=dataset_synthetic,
            ls_cts_features=self._resource_spec.ls_cts_features,
            stat=DescriptiveStatistic.VARIANCE,
        )
        return plot


@TableComparisonPlotRegistry.register_artifact(TableComparisonPlotType.MEDIAN_ALIGNMENT_PLOT)
class MedianAlignmentPlot(TableComparisonPlot[NoArtifactHyperparams]):
    def _compare_datasets(
        self, dataset_real: pd.DataFrame, dataset_synthetic: pd.DataFrame
    ) -> Figure:
        plot = DescriptiveStatsAlignmentPlotter.get_single_stat_alignment_plot(
            dataset_real=dataset_real,
            dataset_synthetic=dataset_synthetic,
            ls_cts_features=self._resource_spec.ls_cts_features,
            stat=DescriptiveStatistic.MEDIAN,
        )
        return plot


@TableComparisonPlotRegistry.register_artifact(
    TableComparisonPlotType.FIRST_QUARTILE_ALIGNMENT_PLOT
)
class FirstQuartileAlignmentPlot(TableComparisonPlot[NoArtifactHyperparams]):
    def _compare_datasets(
        self, dataset_real: pd.DataFrame, dataset_synthetic: pd.DataFrame
    ) -> Figure:
        plot = DescriptiveStatsAlignmentPlotter.get_single_stat_alignment_plot(
            dataset_real=dataset_real,
            dataset_synthetic=dataset_synthetic,
            ls_cts_features=self._resource_spec.ls_cts_features,
            stat=DescriptiveStatistic.Q1,
        )
        return plot


@TableComparisonPlotRegistry.register_artifact(
    TableComparisonPlotType.THIRD_QUARTILE_ALIGNMENT_PLOT
)
class ThirdQuartileAlignmentPlot(TableComparisonPlot[NoArtifactHyperparams]):
    def _compare_datasets(
        self, dataset_real: pd.DataFrame, dataset_synthetic: pd.DataFrame
    ) -> Figure:
        plot = DescriptiveStatsAlignmentPlotter.get_single_stat_alignment_plot(
            dataset_real=dataset_real,
            dataset_synthetic=dataset_synthetic,
            ls_cts_features=self._resource_spec.ls_cts_features,
            stat=DescriptiveStatistic.Q3,
        )
        return plot


@TableComparisonPlotRegistry.register_artifact(TableComparisonPlotType.MAX_ALIGNMENT_PLOT)
class MaxAlignmentPlot(TableComparisonPlot[NoArtifactHyperparams]):
    def _compare_datasets(
        self, dataset_real: pd.DataFrame, dataset_synthetic: pd.DataFrame
    ) -> Figure:
        plot = DescriptiveStatsAlignmentPlotter.get_single_stat_alignment_plot(
            dataset_real=dataset_real,
            dataset_synthetic=dataset_synthetic,
            ls_cts_features=self._resource_spec.ls_cts_features,
            stat=DescriptiveStatistic.MAX,
        )
        return plot


@TableComparisonPlotRegistry.register_artifact(TableComparisonPlotType.MIN_ALIGNMENT_PLOT)
class MinAlignmentPlot(TableComparisonPlot[NoArtifactHyperparams]):
    def _compare_datasets(
        self, dataset_real: pd.DataFrame, dataset_synthetic: pd.DataFrame
    ) -> Figure:
        plot = DescriptiveStatsAlignmentPlotter.get_single_stat_alignment_plot(
            dataset_real=dataset_real,
            dataset_synthetic=dataset_synthetic,
            ls_cts_features=self._resource_spec.ls_cts_features,
            stat=DescriptiveStatistic.MIN,
        )
        return plot
