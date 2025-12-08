from typing import List, Optional

import pandas as pd
import pytest
from artifact_core._libs.tools.plotters.overlaid_cdf_plotter import (
    OverlaidCDFConfig,
    OverlaidCDFPlotter,
)
from matplotlib.figure import Figure


@pytest.mark.unit
@pytest.mark.parametrize(
    "data_a, data_b, feature_name",
    [
        ([1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 3.0, 4.0, 5.0, 6.0], None),
        ([1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 3.0, 4.0, 5.0, 6.0], "my_feature"),
        (list(range(50)), list(range(10, 60)), "comparison"),
    ],
)
def test_plot_overlaid_cdf_returns_figure(
    set_agg_backend: None,
    close_all_figs_after_test: None,
    data_a: List[float],
    data_b: List[float],
    feature_name: Optional[str],
):
    sr_data_a = pd.Series(data_a)
    sr_data_b = pd.Series(data_b)
    result = OverlaidCDFPlotter.plot_overlaid_cdf(
        sr_data_a=sr_data_a, sr_data_b=sr_data_b, feature_name=feature_name
    )
    assert isinstance(result, Figure)


@pytest.mark.unit
@pytest.mark.parametrize(
    "feature_name, expected_title",
    [
        (None, "CDF Comparison"),
        ("my_feature", "CDF Comparison: my_feature"),
        ("test", "CDF Comparison: test"),
    ],
)
def test_plot_overlaid_cdf_title(
    set_agg_backend: None,
    close_all_figs_after_test: None,
    feature_name: Optional[str],
    expected_title: str,
):
    sr_data_a = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    sr_data_b = pd.Series([2.0, 3.0, 4.0, 5.0, 6.0])
    result = OverlaidCDFPlotter.plot_overlaid_cdf(
        sr_data_a=sr_data_a, sr_data_b=sr_data_b, feature_name=feature_name
    )
    title_texts = [t.get_text() for t in result.texts]
    assert expected_title in title_texts


@pytest.mark.unit
def test_plot_overlaid_cdf_with_custom_config(
    set_agg_backend: None, close_all_figs_after_test: None
):
    sr_data_a = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    sr_data_b = pd.Series([2.0, 3.0, 4.0, 5.0, 6.0])
    config = OverlaidCDFConfig(
        plot_color_a="navy",
        plot_color_b="coral",
        label_a="Original",
        label_b="Generated",
        line_width_a=3.0,
        line_width_b=3.0,
    )
    result = OverlaidCDFPlotter.plot_overlaid_cdf(
        sr_data_a=sr_data_a, sr_data_b=sr_data_b, config=config
    )
    assert isinstance(result, Figure)


@pytest.mark.unit
def test_plot_overlaid_cdf_has_legend(set_agg_backend: None, close_all_figs_after_test: None):
    sr_data_a = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    sr_data_b = pd.Series([2.0, 3.0, 4.0, 5.0, 6.0])
    result = OverlaidCDFPlotter.plot_overlaid_cdf(sr_data_a=sr_data_a, sr_data_b=sr_data_b)
    ax = result.axes[0]
    legend = ax.get_legend()
    assert legend is not None


@pytest.mark.unit
def test_plot_overlaid_cdf_has_two_lines(set_agg_backend: None, close_all_figs_after_test: None):
    sr_data_a = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    sr_data_b = pd.Series([2.0, 3.0, 4.0, 5.0, 6.0])
    result = OverlaidCDFPlotter.plot_overlaid_cdf(sr_data_a=sr_data_a, sr_data_b=sr_data_b)
    ax = result.axes[0]
    assert len(ax.lines) == 2
