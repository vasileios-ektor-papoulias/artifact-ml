from typing import List

import pandas as pd
import pytest
from artifact_core.libs.implementation.tabular.cdf.plotter import CDFPlotter
from matplotlib.figure import Figure


@pytest.mark.parametrize(
    "df_dispatcher, cts_features, expected_plot_count",
    [
        (
            "df_simple",
            ["cts_1", "cts_2"],
            2,
        ),
        ("df_simple", ["cts_1"], 1),
        ("df_simple", [], 0),
    ],
    indirect=["df_dispatcher"],
)
def test_get_cdf_plot_collection(
    set_agg_backend,
    close_all_figs_after_test,
    df_dispatcher: pd.DataFrame,
    cts_features: List[str],
    expected_plot_count: int,
):
    df = df_dispatcher
    result = CDFPlotter.get_cdf_plot_collection(
        dataset=df,
        ls_cts_features=cts_features,
    )

    assert isinstance(result, dict), "Result should be a dictionary"
    assert len(result) == expected_plot_count, (
        f"Expected {expected_plot_count} plots, got {len(result)}"
    )

    for feature, fig in result.items():
        assert isinstance(fig, Figure), f"Expected Figure for feature {feature}, got {type(fig)}"
        assert fig.get_axes(), f"Figure for feature {feature} should have at least one axis"
        title_text = fig.texts[0].get_text() if fig.texts else ""
        assert title_text == f"CDF: {feature}", (
            f"Expected title 'CDF: {feature}', got '{title_text}'"
        )


@pytest.mark.parametrize(
    "df_dispatcher, cts_features, expected_axes_count",
    [
        (
            "df_simple",
            ["cts_1", "cts_2"],
            3,
        ),
        ("df_simple", ["cts_1"], 3),
        ("df_simple", [], 0),
    ],
    indirect=["df_dispatcher"],
)
def test_get_cdf_plot(
    set_agg_backend,
    close_all_figs_after_test,
    df_dispatcher: pd.DataFrame,
    cts_features: List[str],
    expected_axes_count: int,
):
    df = df_dispatcher
    result = CDFPlotter.get_cdf_plot(
        dataset=df,
        ls_cts_features=cts_features,
    )

    assert isinstance(result, Figure), "Result should be a Figure"
    if expected_axes_count > 0:
        assert len(result.axes) == expected_axes_count, (
            f"Expected {expected_axes_count} axes, got {len(result.axes)}"
        )
        assert result.texts, "Figure should have title text"
        assert "Cumulative Density Functions" in result.texts[0].get_text(), (
            "Expected title to contain 'Cumulative Density Functions', "
            + "got {result.texts[0].get_text()}"
        )
    else:
        assert len(result.axes) == 0, (
            f"Expected 0 axes for empty features list, got {len(result.axes)}"
        )
