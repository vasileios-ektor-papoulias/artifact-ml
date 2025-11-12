from typing import Type
from unittest.mock import ANY

import pandas as pd
import pytest
from artifact_core._base.primitives import NO_ARTIFACT_HYPERPARAMS
from artifact_core._libs.artifacts.table_comparison.descriptive_stats.alignment_plotter import (
    DescriptiveStatsAlignmentPlotter,
)
from artifact_core._libs.artifacts.table_comparison.descriptive_stats.calculator import (
    DescriptiveStatistic,
)
from artifact_core._libs.resources_spec.tabular.protocol import TabularDataSpecProtocol
from artifact_core.table_comparison._artifacts.base import (
    DatasetComparisonArtifactResources,
    TableComparisonPlot,
)
from artifact_core.table_comparison._artifacts.plots.descriptive_stats import (
    DescriptiveStatsAlignmentPlot,
    FirstQuartileAlignmentPlot,
    MaxAlignmentPlot,
    MeanAlignmentPlot,
    MedianAlignmentPlot,
    MinAlignmentPlot,
    STDAlignmentPlot,
    ThirdQuartileAlignmentPlot,
    VarianceAlignmentPlot,
)
from pytest_mock import MockerFixture


def test_descriptive_stats_comparison_plot(
    mocker: MockerFixture,
    resource_spec: TabularDataSpecProtocol,
    df_real: pd.DataFrame,
    df_synthetic: pd.DataFrame,
):
    fake_plot = Figure()
    patch_get_combined_stat_comparison_plot = mocker.patch.object(
        target=DescriptiveStatsAlignmentPlotter,
        attribute="get_combined_stat_alignment_plot",
        return_value=fake_plot,
    )
    artifact = DescriptiveStatsAlignmentPlot(
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


@pytest.mark.unit
@pytest.mark.parametrize(
    "artifact_class, statistic",
    [
        (MeanAlignmentPlot, DescriptiveStatistic.MEAN),
        (STDAlignmentPlot, DescriptiveStatistic.STD),
        (VarianceAlignmentPlot, DescriptiveStatistic.VARIANCE),
        (MedianAlignmentPlot, DescriptiveStatistic.MEDIAN),
        (FirstQuartileAlignmentPlot, DescriptiveStatistic.Q1),
        (ThirdQuartileAlignmentPlot, DescriptiveStatistic.Q3),
        (MinAlignmentPlot, DescriptiveStatistic.MIN),
        (MaxAlignmentPlot, DescriptiveStatistic.MAX),
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
        DescriptiveStatsAlignmentPlotter,
        "get_stat_alignment_plot",
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
