from typing import List, Optional, Sequence

import pandas as pd
import pytest
from artifact_core._libs.tools.plotters.pmf_plotter import PMFConfig, PMFPlotter
from matplotlib.figure import Figure


@pytest.mark.unit
@pytest.mark.parametrize(
    "data, feature_name, unique_categories",
    [
        (["A", "B", "A", "C", "B"], None, None),
        (["A", "B", "A", "C", "B"], "category_feature", None),
        (["A", "B", "A", "C", "B"], "category_feature", ["A", "B", "C"]),
        (["X", "Y", "Z", "X", "X"], "another", ["X", "Y", "Z"]),
    ],
)
def test_plot_pmf_returns_figure(
    set_agg_backend: None,
    close_all_figs_after_test: None,
    data: List[str],
    feature_name: Optional[str],
    unique_categories: Optional[Sequence[str]],
):
    sr_data = pd.Series(data)
    result = PMFPlotter.plot_pmf(
        sr_data=sr_data, feature_name=feature_name, unique_categories=unique_categories
    )
    assert isinstance(result, Figure)


@pytest.mark.unit
@pytest.mark.parametrize(
    "feature_name, expected_title",
    [
        (None, "PMF"),
        ("my_feature", "PMF: my_feature"),
        ("test", "PMF: test"),
    ],
)
def test_plot_pmf_title(
    set_agg_backend: None,
    close_all_figs_after_test: None,
    feature_name: Optional[str],
    expected_title: str,
):
    sr_data = pd.Series(["A", "B", "A", "C", "B"])
    result = PMFPlotter.plot_pmf(sr_data=sr_data, feature_name=feature_name)
    title_texts = [t.get_text() for t in result.texts]
    assert expected_title in title_texts


@pytest.mark.unit
def test_plot_pmf_with_custom_config(set_agg_backend: None, close_all_figs_after_test: None):
    sr_data = pd.Series(["A", "B", "A", "C", "B"])
    config = PMFConfig(plot_color="blue", alpha=0.5, rotation="horizontal")
    result = PMFPlotter.plot_pmf(sr_data=sr_data, config=config)
    assert isinstance(result, Figure)


@pytest.mark.unit
def test_plot_pmf_preserves_category_order(set_agg_backend: None, close_all_figs_after_test: None):
    sr_data = pd.Series(["C", "A", "B", "A", "C"])
    unique_categories = ["A", "B", "C"]
    result = PMFPlotter.plot_pmf(
        sr_data=sr_data, feature_name="ordered", unique_categories=unique_categories
    )
    ax = result.axes[0]
    xticklabels = [t.get_text() for t in ax.get_xticklabels()]
    assert xticklabels == unique_categories
