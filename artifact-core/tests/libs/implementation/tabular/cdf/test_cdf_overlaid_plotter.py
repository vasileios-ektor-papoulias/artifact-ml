from typing import List, Tuple

import pandas as pd
import pytest
from artifact_core._libs.artifacts.table_comparison.cdf.overlaid_plotter import (
    TabularOverlaidCDFPlotter,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    "df_pair_dispatcher, cts_features, expected_plot_count",
    [
        (("df_small_real", "df_small_synthetic"), ["cts_1", "cts_2"], 2),
        (("df_small_real", "df_small_synthetic"), ["cts_1"], 1),
        (("df_small_real", "df_small_synthetic"), [], 0),
    ],
    indirect=["df_pair_dispatcher"],
)
def test_get_overlaid_cdf_plot_collection(
    set_agg_backend,
    close_all_figs_after_test,
    df_pair_dispatcher: Tuple[pd.DataFrame, pd.DataFrame],
    cts_features: List[str],
    expected_plot_count: int,
):
    df_real, df_synthetic = df_pair_dispatcher
    result = TabularOverlaidCDFPlotter.get_overlaid_cdf_plot_collection(
        dataset_real=df_real,
        dataset_synthetic=df_synthetic,
        cts_features=cts_features,
    )

    assert isinstance(result, dict), "Result should be a dictionary"
    assert len(result) == expected_plot_count, (
        f"Expected {expected_plot_count} plots, got {len(result)}"
    )

    for feature, fig in result.items():
        assert isinstance(fig, Figure), f"Expected Figure for feature {feature}, got {type(fig)}"
        assert fig.get_axes(), f"Figure for feature {feature} should have at least one axis"
        title_text = fig.texts[0].get_text() if fig.texts else ""
        assert title_text == f"CDF Comparison: {feature}", (
            f"Expected title 'CDF Comparison: {feature}', got '{title_text}'"
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "df_pair_dispatcher, cts_features, expected_axes_count",
    [
        (
            ("df_small_real", "df_small_synthetic"),
            ["cts_1", "cts_2"],
            3,
        ),
        (("df_small_real", "df_small_synthetic"), ["cts_1"], 3),
        (("df_small_real", "df_small_synthetic"), [], 0),
    ],
    indirect=["df_pair_dispatcher"],
)
def test_get_overlaid_cdf_plot(
    set_agg_backend,
    close_all_figs_after_test,
    df_pair_dispatcher: Tuple[pd.DataFrame, pd.DataFrame],
    cts_features: List[str],
    expected_axes_count: int,
):
    df_real, df_synthetic = df_pair_dispatcher
    result = TabularOverlaidCDFPlotter.get_overlaid_cdf_plot(
        dataset_real=df_real,
        dataset_synthetic=df_synthetic,
        cts_features=cts_features,
    )

    assert isinstance(result, Figure), "Result should be a Figure"
    if expected_axes_count > 0:
        assert len(result.axes) == expected_axes_count, (
            f"Expected {expected_axes_count} axes, got {len(result.axes)}"
        )
        assert result.texts, "Figure should have title text"
        assert "Cumulative Density Function Comparison" in result.texts[0].get_text(), (
            "Expected title to contain 'Cumulative Density Function Comparison', "
            + "got {result.texts[0].get_text()}"
        )
    else:
        assert len(result.axes) == 0, (
            f"Expected 0 axes for empty features list, got {len(result.axes)}"
        )
