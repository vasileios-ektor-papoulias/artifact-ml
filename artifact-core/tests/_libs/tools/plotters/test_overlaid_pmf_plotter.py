from typing import List, Optional, Sequence

import pandas as pd
import pytest
from artifact_core._libs.tools.plotters.overlaid_pmf_plotter import (
    OverlaidPMFConfig,
    OverlaidPMFPlotter,
)
from matplotlib.figure import Figure


@pytest.mark.unit
@pytest.mark.parametrize(
    "data_a, data_b, feature_name, unique_categories",
    [
        (["A", "B", "A", "C"], ["B", "C", "C", "A"], None, None),
        (["A", "B", "A", "C"], ["B", "C", "C", "A"], "category", None),
        (["A", "B", "A", "C"], ["B", "C", "C", "A"], "category", ["A", "B", "C"]),
        (["X", "Y", "Z"], ["X", "X", "Y"], "test", ["X", "Y", "Z"]),
    ],
)
def test_plot_overlaid_pmf_returns_figure(
    set_agg_backend: None,
    close_all_figs_after_test: None,
    data_a: List[str],
    data_b: List[str],
    feature_name: Optional[str],
    unique_categories: Optional[Sequence[str]],
):
    sr_data_a = pd.Series(data_a)
    sr_data_b = pd.Series(data_b)
    result = OverlaidPMFPlotter.plot_overlaid_pmf(
        sr_data_a=sr_data_a,
        sr_data_b=sr_data_b,
        feature_name=feature_name,
        unique_categories=unique_categories,
    )
    assert isinstance(result, Figure)


@pytest.mark.unit
@pytest.mark.parametrize(
    "feature_name, expected_title",
    [
        (None, "PMF Comparison"),
        ("my_feature", "PMF Comparison: my_feature"),
        ("test", "PMF Comparison: test"),
    ],
)
def test_plot_overlaid_pmf_title(
    set_agg_backend: None,
    close_all_figs_after_test: None,
    feature_name: Optional[str],
    expected_title: str,
):
    sr_data_a = pd.Series(["A", "B", "A", "C"])
    sr_data_b = pd.Series(["B", "C", "C", "A"])
    result = OverlaidPMFPlotter.plot_overlaid_pmf(
        sr_data_a=sr_data_a, sr_data_b=sr_data_b, feature_name=feature_name
    )
    title_texts = [t.get_text() for t in result.texts]
    assert expected_title in title_texts


@pytest.mark.unit
def test_plot_overlaid_pmf_with_custom_config(
    set_agg_backend: None, close_all_figs_after_test: None
):
    sr_data_a = pd.Series(["A", "B", "A", "C"])
    sr_data_b = pd.Series(["B", "C", "C", "A"])
    config = OverlaidPMFConfig(
        plot_color_a="purple",
        plot_color_b="yellow",
        label_a="Real",
        label_b="Synthetic",
        cat_pmf_bar_width=0.3,
    )
    result = OverlaidPMFPlotter.plot_overlaid_pmf(
        sr_data_a=sr_data_a, sr_data_b=sr_data_b, config=config
    )
    assert isinstance(result, Figure)


@pytest.mark.unit
def test_plot_overlaid_pmf_has_legend(set_agg_backend: None, close_all_figs_after_test: None):
    sr_data_a = pd.Series(["A", "B", "A", "C"])
    sr_data_b = pd.Series(["B", "C", "C", "A"])
    result = OverlaidPMFPlotter.plot_overlaid_pmf(sr_data_a=sr_data_a, sr_data_b=sr_data_b)
    ax = result.axes[0]
    legend = ax.get_legend()
    assert legend is not None


@pytest.mark.unit
def test_plot_overlaid_pmf_has_two_bar_groups(
    set_agg_backend: None, close_all_figs_after_test: None
):
    sr_data_a = pd.Series(["A", "B", "C"])
    sr_data_b = pd.Series(["A", "B", "C"])
    result = OverlaidPMFPlotter.plot_overlaid_pmf(
        sr_data_a=sr_data_a, sr_data_b=sr_data_b, unique_categories=["A", "B", "C"]
    )
    ax = result.axes[0]
    assert len(ax.patches) == 6
