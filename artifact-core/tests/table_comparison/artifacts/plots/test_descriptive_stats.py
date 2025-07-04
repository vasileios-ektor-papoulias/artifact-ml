from typing import Type
from unittest.mock import ANY

import pandas as pd
import pytest
from artifact_core.base.artifact_dependencies import NO_ARTIFACT_HYPERPARAMS
from artifact_core.libs.implementation.tabular.descriptive_statistics.calculator import (
    DescriptiveStatistic,
)
from artifact_core.libs.implementation.tabular.descriptive_statistics.comparison_plots import (
    DescriptiveStatComparisonPlotter,
)
from artifact_core.libs.resource_spec.tabular.protocol import TabularDataSpecProtocol
from artifact_core.table_comparison.artifacts.base import (
    DatasetComparisonArtifactResources,
    TableComparisonPlot,
)
from artifact_core.table_comparison.artifacts.plots.descriptive_stats import (
    DescriptiveStatsComparisonPlot,
    FirstQuartileComparisonPlot,
    MaxComparisonPlot,
    MeanComparisonPlot,
    MedianComparisonPlot,
    MinComparisonPlot,
    STDComparisonPlot,
    ThirdQuartileComparisonPlot,
    VarianceComparisonPlot,
)
from matplotlib.figure import Figure
from pytest_mock import MockerFixture


def test_descriptive_stats_comparison_plot(
    mocker: MockerFixture,
    resource_spec: TabularDataSpecProtocol,
    df_real: pd.DataFrame,
    df_synthetic: pd.DataFrame,
):
    fake_plot = Figure()
    patch_get_combined_stat_comparison_plot = mocker.patch.object(
        target=DescriptiveStatComparisonPlotter,
        attribute="get_combined_stat_comparison_plot",
        return_value=fake_plot,
    )
    artifact = DescriptiveStatsComparisonPlot(
        resource_spec=resource_spec, hyperparams=NO_ARTIFACT_HYPERPARAMS
    )
    resources = DatasetComparisonArtifactResources(
        dataset_real=df_real, dataset_synthetic=df_synthetic
    )
    result = artifact.compute(resources=resources)
    patch_get_combined_stat_comparison_plot.assert_called_once_with(
        dataset_real=ANY,
        dataset_synthetic=ANY,
        ls_cts_features=resource_spec.ls_cts_features,
    )
    _, kwargs = patch_get_combined_stat_comparison_plot.call_args
    pd.testing.assert_frame_equal(kwargs["dataset_real"], df_real)
    pd.testing.assert_frame_equal(kwargs["dataset_synthetic"], df_synthetic)
    assert result == fake_plot


@pytest.mark.parametrize(
    "artifact_class, statistic",
    [
        (MeanComparisonPlot, DescriptiveStatistic.MEAN),
        (STDComparisonPlot, DescriptiveStatistic.STD),
        (VarianceComparisonPlot, DescriptiveStatistic.VARIANCE),
        (MedianComparisonPlot, DescriptiveStatistic.MEDIAN),
        (FirstQuartileComparisonPlot, DescriptiveStatistic.Q1),
        (ThirdQuartileComparisonPlot, DescriptiveStatistic.Q3),
        (MinComparisonPlot, DescriptiveStatistic.MIN),
        (MaxComparisonPlot, DescriptiveStatistic.MAX),
    ],
)
def test_stat_comparison_plots(
    mocker: MockerFixture,
    resource_spec: TabularDataSpecProtocol,
    df_real: pd.DataFrame,
    df_synthetic: pd.DataFrame,
    artifact_class: Type[TableComparisonPlot],
    statistic: DescriptiveStatistic,
):
    fake_plot = Figure()
    mock_get_stat_comparison_plot = mocker.patch.object(
        DescriptiveStatComparisonPlotter,
        "get_stat_comparison_plot",
        return_value=fake_plot,
    )
    artifact = artifact_class(resource_spec=resource_spec, hyperparams=NO_ARTIFACT_HYPERPARAMS)
    resources = DatasetComparisonArtifactResources(
        dataset_real=df_real, dataset_synthetic=df_synthetic
    )
    result = artifact.compute(resources=resources)
    mock_get_stat_comparison_plot.assert_called_once_with(
        dataset_real=ANY,
        dataset_synthetic=ANY,
        ls_cts_features=resource_spec.ls_cts_features,
        stat=statistic,
    )
    _, kwargs = mock_get_stat_comparison_plot.call_args
    pd.testing.assert_frame_equal(kwargs["dataset_real"], df_real)
    pd.testing.assert_frame_equal(kwargs["dataset_synthetic"], df_synthetic)
    assert result == fake_plot
