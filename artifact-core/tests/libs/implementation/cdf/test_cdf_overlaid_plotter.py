from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from artifact_core.libs.implementation.cdf.overlaid_plotter import OverlaidCDFPlotter
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
            "continuous_feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "continuous_feature2": [5.0, 4.0, 3.0, 2.0, 1.0],
            "unused_feature": ["X", "Y", "Z", "X", "Y"],
        }
    )


@pytest.fixture
def df_synthetic() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "continuous_feature1": [1.5, 2.5, 3.5, 4.5, 5.5],
            "continuous_feature2": [5.5, 4.5, 3.5, 2.5, 1.5],
            "unused_feature": ["Y", "X", "Y", "Z", "X"],
        }
    )


@pytest.mark.parametrize(
    "cts_features, expected_plot_count",
    [
        (["continuous_feature1", "continuous_feature2"], 2),
        (["continuous_feature1"], 1),
        ([], 0),
    ],
)
def test_get_overlaid_cdf_plot_collection(
    set_agg_backend,
    close_all_figs_after_test,
    df_real: pd.DataFrame,
    df_synthetic: pd.DataFrame,
    cts_features: List[str],
    expected_plot_count: int,
):
    result = OverlaidCDFPlotter.get_overlaid_cdf_plot_collection(
        dataset_real=df_real,
        dataset_synthetic=df_synthetic,
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
        assert title_text == f"CDF Comparison: {feature}", (
            f"Expected title 'CDF Comparison: {feature}', got '{title_text}'"
        )


@pytest.mark.parametrize(
    "cts_features, expected_axes_count",
    [
        (
            ["continuous_feature1", "continuous_feature2"],
            3,
        ),  # 2 features with 3 cols = 3 axes (1 will be empty)
        (["continuous_feature1"], 3),  # 1 feature with 3 cols = 3 axes (2 will be empty)
        ([], 0),  # No features = no axes
    ],
)
def test_get_overlaid_cdf_plot(
    set_agg_backend,
    close_all_figs_after_test,
    df_real: pd.DataFrame,
    df_synthetic: pd.DataFrame,
    cts_features: List[str],
    expected_axes_count: int,
):
    result = OverlaidCDFPlotter.get_overlaid_cdf_plot(
        dataset_real=df_real,
        dataset_synthetic=df_synthetic,
        ls_cts_features=cts_features,
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
