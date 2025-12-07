from typing import List, Optional

import pandas as pd
import pytest
from artifact_core._libs.tools.plotters.cdf_plotter import CDFConfig, CDFPlotter
from matplotlib.figure import Figure


@pytest.mark.unit
@pytest.mark.parametrize(
    "data, feature_name",
    [
        ([1.0, 2.0, 3.0, 4.0, 5.0], None),
        ([1.0, 2.0, 3.0, 4.0, 5.0], "my_feature"),
        ([5.0, 3.0, 1.0, 4.0, 2.0], "unsorted"),
        (list(range(100)), "large_data"),
    ],
)
def test_plot_cdf_returns_figure(
    set_agg_backend,
    close_all_figs_after_test,
    data: List[float],
    feature_name: Optional[str],
):
    sr_data = pd.Series(data)
    result = CDFPlotter.plot_cdf(sr_data=sr_data, feature_name=feature_name)
    assert isinstance(result, Figure)


@pytest.mark.unit
@pytest.mark.parametrize(
    "feature_name, expected_title",
    [
        (None, "CDF"),
        ("my_feature", "CDF: my_feature"),
        ("test", "CDF: test"),
    ],
)
def test_plot_cdf_title(
    set_agg_backend,
    close_all_figs_after_test,
    feature_name: Optional[str],
    expected_title: str,
):
    sr_data = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = CDFPlotter.plot_cdf(sr_data=sr_data, feature_name=feature_name)
    title_texts = [t.get_text() for t in result.texts]
    assert expected_title in title_texts


@pytest.mark.unit
def test_plot_cdf_with_custom_config(set_agg_backend, close_all_figs_after_test):
    sr_data = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    config = CDFConfig(plot_color="green", line_width=2.0, plot_marker="x")
    result = CDFPlotter.plot_cdf(sr_data=sr_data, config=config)
    assert isinstance(result, Figure)


@pytest.mark.unit
def test_plot_cdf_has_one_line(set_agg_backend, close_all_figs_after_test):
    sr_data = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = CDFPlotter.plot_cdf(sr_data=sr_data)
    ax = result.axes[0]
    assert len(ax.lines) == 1
