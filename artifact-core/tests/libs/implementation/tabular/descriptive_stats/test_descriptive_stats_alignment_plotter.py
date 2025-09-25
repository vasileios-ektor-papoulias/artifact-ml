from typing import List, Tuple

import pandas as pd
import pytest
from artifact_core.libs.implementation.tabular.descriptive_stats.alignment_plotter import (
    DescriptiveStatistic,
    DescriptiveStatsAlignmentPlotter,
)
from matplotlib.figure import Figure


@pytest.mark.unit
@pytest.mark.parametrize(
    "df_pair_dispatcher, ls_cts_features, stat, "
    + "expected_title, expected_ax_title, expected_xlabel, expected_ylabel",
    [
        (
            ("df_small_real", "df_small_synthetic"),
            ["cts_1", "cts_2"],
            DescriptiveStatistic.STD,
            "Log-Abs Std of Continuous Features",
            "std comparison",
            "real data std (log-abs)",
            "synthetic data std (log-abs)",
        ),
        (
            ("df_large_real", "df_large_synthetic"),
            ["cts_1", "cts_2"],
            DescriptiveStatistic.STD,
            "Log-Abs Std of Continuous Features",
            "std comparison",
            "real data std (log-abs)",
            "synthetic data std (log-abs)",
        ),
        (
            ("df_small_real", "df_small_synthetic"),
            ["cts_1", "cts_2"],
            DescriptiveStatistic.VARIANCE,
            "Log-Abs Variance of Continuous Features",
            "variance comparison",
            "real data variance (log-abs)",
            "synthetic data variance (log-abs)",
        ),
        (
            ("df_large_real", "df_large_synthetic"),
            ["cts_1", "cts_2"],
            DescriptiveStatistic.VARIANCE,
            "Log-Abs Variance of Continuous Features",
            "variance comparison",
            "real data variance (log-abs)",
            "synthetic data variance (log-abs)",
        ),
        (
            ("df_small_real", "df_small_synthetic"),
            ["cts_1", "cts_2"],
            DescriptiveStatistic.MEDIAN,
            "Log-Abs Median of Continuous Features",
            "median comparison",
            "real data median (log-abs)",
            "synthetic data median (log-abs)",
        ),
        (
            ("df_large_real", "df_large_synthetic"),
            ["cts_1", "cts_2"],
            DescriptiveStatistic.MEDIAN,
            "Log-Abs Median of Continuous Features",
            "median comparison",
            "real data median (log-abs)",
            "synthetic data median (log-abs)",
        ),
        (
            ("df_small_real", "df_small_synthetic"),
            ["cts_1", "cts_2"],
            DescriptiveStatistic.Q1,
            "Log-Abs Q1 of Continuous Features",
            "q1 comparison",
            "real data q1 (log-abs)",
            "synthetic data q1 (log-abs)",
        ),
        (
            ("df_large_real", "df_large_synthetic"),
            ["cts_1", "cts_2"],
            DescriptiveStatistic.Q1,
            "Log-Abs Q1 of Continuous Features",
            "q1 comparison",
            "real data q1 (log-abs)",
            "synthetic data q1 (log-abs)",
        ),
        (
            ("df_small_real", "df_small_synthetic"),
            ["cts_1", "cts_2"],
            DescriptiveStatistic.Q3,
            "Log-Abs Q3 of Continuous Features",
            "q3 comparison",
            "real data q3 (log-abs)",
            "synthetic data q3 (log-abs)",
        ),
        (
            ("df_large_real", "df_large_synthetic"),
            ["cts_1", "cts_2"],
            DescriptiveStatistic.Q3,
            "Log-Abs Q3 of Continuous Features",
            "q3 comparison",
            "real data q3 (log-abs)",
            "synthetic data q3 (log-abs)",
        ),
        (
            ("df_small_real", "df_small_synthetic"),
            ["cts_1", "cts_2"],
            DescriptiveStatistic.MIN,
            "Log-Abs Min of Continuous Features",
            "min comparison",
            "real data min (log-abs)",
            "synthetic data min (log-abs)",
        ),
        (
            ("df_large_real", "df_large_synthetic"),
            ["cts_1", "cts_2"],
            DescriptiveStatistic.MIN,
            "Log-Abs Min of Continuous Features",
            "min comparison",
            "real data min (log-abs)",
            "synthetic data min (log-abs)",
        ),
        (
            ("df_small_real", "df_small_synthetic"),
            ["cts_1", "cts_2"],
            DescriptiveStatistic.MAX,
            "Log-Abs Max of Continuous Features",
            "max comparison",
            "real data max (log-abs)",
            "synthetic data max (log-abs)",
        ),
        (
            ("df_large_real", "df_large_synthetic"),
            ["cts_1", "cts_2"],
            DescriptiveStatistic.MAX,
            "Log-Abs Max of Continuous Features",
            "max comparison",
            "real data max (log-abs)",
            "synthetic data max (log-abs)",
        ),
    ],
    indirect=["df_pair_dispatcher"],
)
def test_get_stat_comparison_plot(
    set_agg_backend,
    close_all_figs_after_test,
    df_pair_dispatcher: Tuple[pd.DataFrame, pd.DataFrame],
    ls_cts_features: List[str],
    stat: DescriptiveStatistic,
    expected_title: str,
    expected_ax_title: str,
    expected_xlabel: str,
    expected_ylabel: str,
):
    df_real, df_synthetic = df_pair_dispatcher
    result = DescriptiveStatsAlignmentPlotter.get_stat_alignment_plot(
        dataset_real=df_real,
        dataset_synthetic=df_synthetic,
        ls_cts_features=ls_cts_features,
        stat=stat,
    )
    assert isinstance(result, Figure), "Result should be a Figure"
    assert len(result.texts) >= 1, "Figure should have at least one text element (title)"
    title = result.texts[0]
    assert expected_title in title.get_text(), (
        f"Expected title to contain '{expected_title}', got '{title.get_text()}'"
    )
    assert result.get_axes(), "Figure should have at least one axis"
    ls_axes = result.get_axes()
    assert len(ls_axes) == 1, "Figure should have precisely one axis"
    ax = result.axes[0]
    ax_title = ax.get_title()
    assert ax_title == expected_ax_title, (
        f"Expected ax title '{expected_ax_title}', got '{ax_title}'"
    )
    xlabel = ax.get_xlabel()
    assert xlabel == expected_xlabel, f"Expected xlabel '{expected_xlabel}', got '{xlabel}'"
    ylabel = ax.get_ylabel()
    assert ylabel == expected_ylabel, f"Expected ylabel '{expected_ylabel}', got '{ylabel}'"
    assert ax.get_xticklabels(), "Axis should have x tick labels"
    assert ax.get_yticklabels(), "Axis should have y tick labels"


@pytest.mark.unit
@pytest.mark.parametrize(
    "df_pair_dispatcher, ls_cts_features, expected_title",
    [
        (
            ("df_small_real", "df_small_synthetic"),
            ["cts_1", "cts_2"],
            "Descriptive Statistics Comparison",
        ),
        (
            ("df_large_real", "df_large_synthetic"),
            ["cts_1", "cts_2"],
            "Descriptive Statistics Comparison",
        ),
    ],
    indirect=["df_pair_dispatcher"],
)
def test_get_combined_stat_comparison_plot(
    set_agg_backend,
    close_all_figs_after_test,
    df_pair_dispatcher: Tuple[pd.DataFrame, pd.DataFrame],
    ls_cts_features: List[str],
    expected_title: str,
):
    df_real, df_synthetic = df_pair_dispatcher
    result = DescriptiveStatsAlignmentPlotter.get_combined_stat_alignment_plot(
        dataset_real=df_real,
        dataset_synthetic=df_synthetic,
        ls_cts_features=ls_cts_features,
    )
    assert isinstance(result, Figure), "Result should be a Figure"
    assert result.get_axes(), "Figure should have at least one axis"
    assert len(result.texts) >= 1, "Figure should have at least one text element (title)"
    title_text = result.texts[0].get_text() if result.texts else ""
    assert expected_title in title_text, (
        f"Expected title to contain 'Descriptive Statistics Comparison', got '{title_text}'"
    )
    expected_stats_count = len([stat for stat in DescriptiveStatistic])
    assert len(result.axes) >= expected_stats_count, (
        f"Expected at least {expected_stats_count} axes, got {len(result.axes)}"
    )
    for i, ax in enumerate(result.axes[:expected_stats_count]):
        assert ax is not None, f"Subplot {i} should exist"
