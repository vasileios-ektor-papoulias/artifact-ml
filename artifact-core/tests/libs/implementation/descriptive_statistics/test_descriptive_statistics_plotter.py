from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from artifact_core.libs.implementation.descriptive_statistics.comparison_plots import (
    DescriptiveStatComparisonPlotter,
    DescriptiveStatistic,
)
from matplotlib.figure import Figure


@pytest.fixture
def set_agg_backend():
    import matplotlib

    matplotlib.use("Agg")


@pytest.fixture
def close_all_figs_after_test():
    yield
    plt.close("all")


@pytest.fixture
def df_real() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "num1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "num2": [5.0, 4.0, 3.0, 2.0, 1.0],
            "cat1": ["A", "B", "A", "C", "B"],
            "cat2": ["X", "Y", "X", "Z", "Y"],
        }
    )


@pytest.fixture
def df_synthetic() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "num1": [1.5, 2.5, 3.5, 4.5, 5.5],
            "num2": [4.5, 3.5, 2.5, 1.5, 0.5],
            "cat1": ["B", "A", "C", "B", "A"],
            "cat2": ["Y", "X", "Z", "Y", "X"],
        }
    )


@pytest.fixture
def ls_cts_features() -> List[str]:
    return ["num1", "num2"]


@pytest.mark.parametrize(
    "stat, expected_title, expected_ax_title, expected_xlabel, expected_ylabel",
    [
        (
            DescriptiveStatistic.STD,
            "Log-Abs Std of Continuous Features",
            "std comparison",
            "real data std (log-abs)",
            "synthetic data std (log-abs)",
        ),
        (
            DescriptiveStatistic.VARIANCE,
            "Log-Abs Variance of Continuous Features",
            "variance comparison",
            "real data variance (log-abs)",
            "synthetic data variance (log-abs)",
        ),
        (
            DescriptiveStatistic.MEDIAN,
            "Log-Abs Median of Continuous Features",
            "median comparison",
            "real data median (log-abs)",
            "synthetic data median (log-abs)",
        ),
        (
            DescriptiveStatistic.Q1,
            "Log-Abs Q1 of Continuous Features",
            "q1 comparison",
            "real data q1 (log-abs)",
            "synthetic data q1 (log-abs)",
        ),
        (
            DescriptiveStatistic.Q3,
            "Log-Abs Q3 of Continuous Features",
            "q3 comparison",
            "real data q3 (log-abs)",
            "synthetic data q3 (log-abs)",
        ),
        (
            DescriptiveStatistic.MIN,
            "Log-Abs Min of Continuous Features",
            "min comparison",
            "real data min (log-abs)",
            "synthetic data min (log-abs)",
        ),
        (
            DescriptiveStatistic.MAX,
            "Log-Abs Max of Continuous Features",
            "max comparison",
            "real data max (log-abs)",
            "synthetic data max (log-abs)",
        ),
    ],
)
def test_get_stat_comparison_plot(
    set_agg_backend,
    close_all_figs_after_test,
    df_real: pd.DataFrame,
    df_synthetic: pd.DataFrame,
    ls_cts_features: List[str],
    stat: DescriptiveStatistic,
    expected_title: str,
    expected_ax_title: str,
    expected_xlabel: str,
    expected_ylabel: str,
):
    result = DescriptiveStatComparisonPlotter.get_stat_comparison_plot(
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


def test_get_combined_stat_comparison_plot(
    set_agg_backend,
    close_all_figs_after_test,
    df_real: pd.DataFrame,
    df_synthetic: pd.DataFrame,
    ls_cts_features: List[str],
):
    result = DescriptiveStatComparisonPlotter.get_combined_stat_comparison_plot(
        dataset_real=df_real,
        dataset_synthetic=df_synthetic,
        ls_cts_features=ls_cts_features,
    )
    assert isinstance(result, Figure), "Result should be a Figure"
    assert result.get_axes(), "Figure should have at least one axis"
    assert len(result.texts) >= 1, "Figure should have at least one text element (title)"
    title_text = result.texts[0].get_text() if result.texts else ""
    assert "Descriptive Statistics Comparison" in title_text, (
        f"Expected title to contain 'Descriptive Statistics Comparison', got '{title_text}'"
    )
    expected_stats_count = len([stat for stat in DescriptiveStatistic])
    assert len(result.axes) >= expected_stats_count, (
        f"Expected at least {expected_stats_count} axes, got {len(result.axes)}"
    )
    for i, ax in enumerate(result.axes[:expected_stats_count]):
        assert ax is not None, f"Subplot {i} should exist"
