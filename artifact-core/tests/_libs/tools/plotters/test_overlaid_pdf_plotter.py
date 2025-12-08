from typing import List, Optional

import pandas as pd
import pytest
from artifact_core._libs.tools.plotters.overlaid_pdf_plotter import (
    OverlaidPDFConfig,
    OverlaidPDFPlotter,
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
def test_plot_overlaid_pdf_returns_figure(
    set_agg_backend: None,
    close_all_figs_after_test: None,
    data_a: List[float],
    data_b: List[float],
    feature_name: Optional[str],
):
    sr_data_a = pd.Series(data_a)
    sr_data_b = pd.Series(data_b)
    result = OverlaidPDFPlotter.plot_overlaid_pdf(
        sr_data_a=sr_data_a, sr_data_b=sr_data_b, feature_name=feature_name
    )
    assert isinstance(result, Figure)


@pytest.mark.unit
@pytest.mark.parametrize(
    "feature_name, expected_title",
    [
        (None, "PDF Comparison"),
        ("my_feature", "PDF Comparison: my_feature"),
        ("test", "PDF Comparison: test"),
    ],
)
def test_plot_overlaid_pdf_title(
    set_agg_backend: None,
    close_all_figs_after_test: None,
    feature_name: Optional[str],
    expected_title: str,
):
    sr_data_a = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    sr_data_b = pd.Series([2.0, 3.0, 4.0, 5.0, 6.0])
    result = OverlaidPDFPlotter.plot_overlaid_pdf(
        sr_data_a=sr_data_a, sr_data_b=sr_data_b, feature_name=feature_name
    )
    title_texts = [t.get_text() for t in result.texts]
    assert expected_title in title_texts


@pytest.mark.unit
def test_plot_overlaid_pdf_with_custom_config(
    set_agg_backend: None, close_all_figs_after_test: None
):
    sr_data_a = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    sr_data_b = pd.Series([2.0, 3.0, 4.0, 5.0, 6.0])
    config = OverlaidPDFConfig(
        plot_color_a="blue",
        plot_color_b="orange",
        label_a="Dataset A",
        label_b="Dataset B",
    )
    result = OverlaidPDFPlotter.plot_overlaid_pdf(
        sr_data_a=sr_data_a, sr_data_b=sr_data_b, config=config
    )
    assert isinstance(result, Figure)


@pytest.mark.unit
def test_plot_overlaid_pdf_has_legend(set_agg_backend: None, close_all_figs_after_test: None):
    sr_data_a = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    sr_data_b = pd.Series([2.0, 3.0, 4.0, 5.0, 6.0])
    result = OverlaidPDFPlotter.plot_overlaid_pdf(sr_data_a=sr_data_a, sr_data_b=sr_data_b)
    ax = result.axes[0]
    legend = ax.get_legend()
    assert legend is not None
