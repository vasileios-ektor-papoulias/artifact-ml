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
def sample_dataframe_real():
    return pd.DataFrame(
        {
            "continuous_feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "continuous_feature2": [5.0, 4.0, 3.0, 2.0, 1.0],
            "unused_feature": ["X", "Y", "Z", "X", "Y"],
        }
    )


@pytest.fixture
def sample_dataframe_synthetic():
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
    sample_dataframe_real,
    sample_dataframe_synthetic,
    cts_features: List[str],
    expected_plot_count: int,
):
    result = OverlaidCDFPlotter.get_overlaid_cdf_plot_collection(
        dataset_real=sample_dataframe_real,
        dataset_synthetic=sample_dataframe_synthetic,
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
    sample_dataframe_real,
    sample_dataframe_synthetic,
    cts_features: List[str],
    expected_axes_count: int,
):
    result = OverlaidCDFPlotter.get_overlaid_cdf_plot(
        dataset_real=sample_dataframe_real,
        dataset_synthetic=sample_dataframe_synthetic,
        ls_cts_features=cts_features,
    )

    assert isinstance(result, Figure), "Result should be a Figure"
    if expected_axes_count > 0:
        assert len(result.axes) == expected_axes_count, (
            f"Expected {expected_axes_count} axes, got {len(result.axes)}"
        )
        assert result.texts, "Figure should have title text"
        assert "Cumulative Density Function Comparison" in result.texts[0].get_text(), (
            f"Expected title to contain 'Cumulative Density Function Comparison', got {result.texts[0].get_text()}"
        )
    else:
        assert len(result.axes) == 0, (
            f"Expected 0 axes for empty features list, got {len(result.axes)}"
        )


@pytest.mark.parametrize(
    "data_real, data_synthetic, feature_name, expected_title",
    [
        (
            pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], name="continuous"),
            pd.Series([1.5, 2.5, 3.5, 4.5, 5.5], name="continuous"),
            "continuous",
            "CDF Comparison: continuous",
        ),
    ],
)
def test_plot_overlaid_cdf(
    set_agg_backend,
    close_all_figs_after_test,
    data_real: pd.Series,
    data_synthetic: pd.Series,
    feature_name: str,
    expected_title: str,
):
    result = OverlaidCDFPlotter._plot_overlaid_cdf(
        sr_data_real=data_real, sr_data_synthetic=data_synthetic, feature_name=feature_name
    )
    assert isinstance(result, Figure), "Result should be a Figure"
    assert result.get_axes(), "Figure should have at least one axis"
    title_text = result.texts[0].get_text() if result.texts else ""
    assert title_text == expected_title, f"Expected title '{expected_title}', got '{title_text}'"
    ax = result.get_axes()[0]
    assert ax.get_xlabel() == "Values", f"Expected x-label 'Values', got '{ax.get_xlabel()}'"
    assert ax.get_ylabel() == "Normalized Cumulative Sum", (
        f"Expected y-label 'Normalized Cumulative Sum', got '{ax.get_ylabel()}'"
    )
    lines = ax.get_lines()
    assert len(lines) >= 2, "Expected at least 2 lines (real and synthetic) in the plot"
    for i, line in enumerate(lines[:2]):
        assert hasattr(line, "get_xdata"), f"Line {i} should have get_xdata method"
        assert hasattr(line, "get_ydata"), f"Line {i} should have get_ydata method"
        assert line.get_xdata() is not None, f"Line {i} should have x data"
        assert line.get_ydata() is not None, f"Line {i} should have y data"
    legend_texts = [text.get_text() for text in ax.get_legend().get_texts()]
    assert "Real" in legend_texts, "Legend should contain 'Real'"
    assert "Synthetic" in legend_texts, "Legend should contain 'Synthetic'"
