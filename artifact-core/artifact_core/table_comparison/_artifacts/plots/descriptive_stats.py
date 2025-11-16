import pandas as pd

from artifact_core._base.core.hyperparams import NoArtifactHyperparams
from artifact_core._base.typing.artifact_result import Plot
from artifact_core._libs.artifacts.table_comparison.descriptive_stats.alignment_plotter import (
    DescriptiveStatsAlignmentPlotter,
)
from artifact_core._libs.artifacts.table_comparison.descriptive_stats.calculator import (
    DescriptiveStatistic,
)
from artifact_core.table_comparison._artifacts.base import (
    TableComparisonPlot,
)
from artifact_core.table_comparison._registries.plots import TableComparisonPlotRegistry
from artifact_core.table_comparison._types.plots import TableComparisonPlotType


@TableComparisonPlotRegistry.register_artifact(TableComparisonPlotType.DESCRIPTIVE_STATS_ALIGNMENT)
class DescriptiveStatsAlignmentPlot(TableComparisonPlot[NoArtifactHyperparams]):
    def _compare_datasets(
        self, dataset_real: pd.DataFrame, dataset_synthetic: pd.DataFrame
    ) -> Plot:
        plot = DescriptiveStatsAlignmentPlotter.get_combined_stat_alignment_plot(
            dataset_real=dataset_real,
            dataset_synthetic=dataset_synthetic,
            cts_features=self._resource_spec.cts_features,
        )
        return plot


@TableComparisonPlotRegistry.register_artifact(TableComparisonPlotType.MEAN_ALIGNMENT)
class MeanAlignmentPlot(TableComparisonPlot[NoArtifactHyperparams]):
    def _compare_datasets(
        self, dataset_real: pd.DataFrame, dataset_synthetic: pd.DataFrame
    ) -> Plot:
        plot = DescriptiveStatsAlignmentPlotter.get_stat_alignment_plot(
            dataset_real=dataset_real,
            dataset_synthetic=dataset_synthetic,
            cts_features=self._resource_spec.cts_features,
            stat=DescriptiveStatistic.MEAN,
        )
        return plot


@TableComparisonPlotRegistry.register_artifact(TableComparisonPlotType.STD_ALIGNMENT)
class STDAlignmentPlot(TableComparisonPlot[NoArtifactHyperparams]):
    def _compare_datasets(
        self, dataset_real: pd.DataFrame, dataset_synthetic: pd.DataFrame
    ) -> Plot:
        plot = DescriptiveStatsAlignmentPlotter.get_stat_alignment_plot(
            dataset_real=dataset_real,
            dataset_synthetic=dataset_synthetic,
            cts_features=self._resource_spec.cts_features,
            stat=DescriptiveStatistic.STD,
        )
        return plot


@TableComparisonPlotRegistry.register_artifact(TableComparisonPlotType.VARIANCE_ALIGNMENT)
class VarianceAlignmentPlot(TableComparisonPlot[NoArtifactHyperparams]):
    def _compare_datasets(
        self, dataset_real: pd.DataFrame, dataset_synthetic: pd.DataFrame
    ) -> Plot:
        plot = DescriptiveStatsAlignmentPlotter.get_stat_alignment_plot(
            dataset_real=dataset_real,
            dataset_synthetic=dataset_synthetic,
            cts_features=self._resource_spec.cts_features,
            stat=DescriptiveStatistic.VARIANCE,
        )
        return plot


@TableComparisonPlotRegistry.register_artifact(TableComparisonPlotType.MEDIAN_ALIGNMENT)
class MedianAlignmentPlot(TableComparisonPlot[NoArtifactHyperparams]):
    def _compare_datasets(
        self, dataset_real: pd.DataFrame, dataset_synthetic: pd.DataFrame
    ) -> Plot:
        plot = DescriptiveStatsAlignmentPlotter.get_stat_alignment_plot(
            dataset_real=dataset_real,
            dataset_synthetic=dataset_synthetic,
            cts_features=self._resource_spec.cts_features,
            stat=DescriptiveStatistic.MEDIAN,
        )
        return plot


@TableComparisonPlotRegistry.register_artifact(TableComparisonPlotType.FIRST_QUARTILE_ALIGNMENT)
class FirstQuartileAlignmentPlot(TableComparisonPlot[NoArtifactHyperparams]):
    def _compare_datasets(
        self, dataset_real: pd.DataFrame, dataset_synthetic: pd.DataFrame
    ) -> Plot:
        plot = DescriptiveStatsAlignmentPlotter.get_stat_alignment_plot(
            dataset_real=dataset_real,
            dataset_synthetic=dataset_synthetic,
            cts_features=self._resource_spec.cts_features,
            stat=DescriptiveStatistic.Q1,
        )
        return plot


@TableComparisonPlotRegistry.register_artifact(TableComparisonPlotType.THIRD_QUARTILE_ALIGNMENT)
class ThirdQuartileAlignmentPlot(TableComparisonPlot[NoArtifactHyperparams]):
    def _compare_datasets(
        self, dataset_real: pd.DataFrame, dataset_synthetic: pd.DataFrame
    ) -> Plot:
        plot = DescriptiveStatsAlignmentPlotter.get_stat_alignment_plot(
            dataset_real=dataset_real,
            dataset_synthetic=dataset_synthetic,
            cts_features=self._resource_spec.cts_features,
            stat=DescriptiveStatistic.Q3,
        )
        return plot


@TableComparisonPlotRegistry.register_artifact(TableComparisonPlotType.MAX_ALIGNMENT)
class MaxAlignmentPlot(TableComparisonPlot[NoArtifactHyperparams]):
    def _compare_datasets(
        self, dataset_real: pd.DataFrame, dataset_synthetic: pd.DataFrame
    ) -> Plot:
        plot = DescriptiveStatsAlignmentPlotter.get_stat_alignment_plot(
            dataset_real=dataset_real,
            dataset_synthetic=dataset_synthetic,
            cts_features=self._resource_spec.cts_features,
            stat=DescriptiveStatistic.MAX,
        )
        return plot


@TableComparisonPlotRegistry.register_artifact(TableComparisonPlotType.MIN_ALIGNMENT)
class MinAlignmentPlot(TableComparisonPlot[NoArtifactHyperparams]):
    def _compare_datasets(
        self, dataset_real: pd.DataFrame, dataset_synthetic: pd.DataFrame
    ) -> Plot:
        plot = DescriptiveStatsAlignmentPlotter.get_stat_alignment_plot(
            dataset_real=dataset_real,
            dataset_synthetic=dataset_synthetic,
            cts_features=self._resource_spec.cts_features,
            stat=DescriptiveStatistic.MIN,
        )
        return plot
