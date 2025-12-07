from typing import List, Optional

import pandas as pd
import pytest
from artifact_core._libs.tools.plotters.pdf_plotter import PDFConfig, PDFPlotter
from matplotlib.figure import Figure


@pytest.mark.unit
@pytest.mark.parametrize(
    "data, feature_name",
    [
        ([1.0, 2.0, 3.0, 4.0, 5.0], None),
        ([1.0, 2.0, 3.0, 4.0, 5.0], "my_feature"),
        ([1.0, 1.0, 2.0, 2.0, 3.0, 3.0], "repeated_values"),
        (list(range(100)), "large_data"),
    ],
)
def test_plot_pdf_returns_figure(
    set_agg_backend,
    close_all_figs_after_test,
    data: List[float],
    feature_name: Optional[str],
):
    sr_data = pd.Series(data)
    result = PDFPlotter.plot_pdf(sr_data=sr_data, feature_name=feature_name)
    assert isinstance(result, Figure)


@pytest.mark.unit
@pytest.mark.parametrize(
    "feature_name, expected_title",
    [
        (None, "PDF"),
        ("my_feature", "PDF: my_feature"),
        ("test", "PDF: test"),
    ],
)
def test_plot_pdf_title(
    set_agg_backend,
    close_all_figs_after_test,
    feature_name: Optional[str],
    expected_title: str,
):
    sr_data = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = PDFPlotter.plot_pdf(sr_data=sr_data, feature_name=feature_name)
    title_texts = [t.get_text() for t in result.texts]
    assert expected_title in title_texts


@pytest.mark.unit
def test_plot_pdf_with_custom_config(set_agg_backend, close_all_figs_after_test):
    sr_data = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    config = PDFConfig(
        plot_color="red",
        cts_density_n_bins=10,
        cts_density_enable_kde=False,
    )
    result = PDFPlotter.plot_pdf(sr_data=sr_data, config=config)
    assert isinstance(result, Figure)


@pytest.mark.unit
def test_plot_pdf_with_nan_values(set_agg_backend, close_all_figs_after_test):
    sr_data = pd.Series([1.0, 2.0, float("nan"), 4.0, 5.0])
    result = PDFPlotter.plot_pdf(sr_data=sr_data, feature_name="with_nan")
    assert isinstance(result, Figure)
