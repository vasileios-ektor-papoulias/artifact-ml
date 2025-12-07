from typing import Dict, List

import pandas as pd
import pytest
from artifact_core._libs.artifacts.table_comparison.pdf.plotter import (
    TabularPDFPlotter,
)
from artifact_core._libs.tools.plotters.pdf_plotter import PDFPlotter
from artifact_core._libs.tools.plotters.plot_combiner import PlotCombiner
from artifact_core._libs.tools.plotters.pmf_plotter import PMFPlotter
from matplotlib.figure import Figure
from pytest_mock import MockerFixture


@pytest.mark.unit
@pytest.mark.parametrize(
    "df_dispatcher, cat_unique_map, features_order, cts_features, cat_features, "
    + "expected_plot_count, expected_pdf_calls, expected_pmf_calls",
    [
        (
            "df_simple",
            {"cat_1": ["A", "B", "C"], "cat_2": ["X", "Y", "Z"]},
            ["cts_1", "cts_2", "cat_1", "cat_2"],
            ["cts_1", "cts_2"],
            ["cat_1", "cat_2"],
            4,
            2,
            2,
        ),
        (
            "df_simple",
            {"cat_1": ["A", "B", "C"], "cat_2": ["X", "Y", "Z"]},
            ["cat_1", "cat_2"],
            [],
            ["cat_1", "cat_2"],
            2,
            0,
            2,
        ),
        (
            "df_simple",
            {},
            ["cts_1", "cts_2"],
            ["cts_1", "cts_2"],
            [],
            2,
            2,
            0,
        ),
        (
            "df_simple",
            {},
            [],
            [],
            [],
            0,
            0,
            0,
        ),
    ],
    indirect=["df_dispatcher"],
)
def test_get_pdf_plot_collection(
    mocker: MockerFixture,
    set_agg_backend,
    close_all_figs_after_test,
    df_dispatcher: pd.DataFrame,
    cat_unique_map: Dict[str, List[str]],
    features_order: List[str],
    cts_features: List[str],
    cat_features: List[str],
    expected_plot_count: int,
    expected_pdf_calls: int,
    expected_pmf_calls: int,
):
    spy_pdf = mocker.spy(obj=PDFPlotter, name="plot_pdf")
    spy_pmf = mocker.spy(obj=PMFPlotter, name="plot_pmf")

    result = TabularPDFPlotter.get_pdf_plot_collection(
        dataset=df_dispatcher,
        features_order=features_order,
        cts_features=cts_features,
        cat_features=cat_features,
        cat_unique_map=cat_unique_map,
    )
    assert isinstance(result, dict)
    assert len(result) == expected_plot_count
    assert spy_pdf.call_count == expected_pdf_calls
    assert spy_pmf.call_count == expected_pmf_calls
    for feature, fig in result.items():
        assert isinstance(fig, Figure)
        assert fig.get_axes()
        title_text = fig.texts[0].get_text() if fig.texts else ""
        assert title_text in (f"PMF: {feature}", f"PDF: {feature}")


@pytest.mark.unit
@pytest.mark.parametrize(
    "df_dispatcher, cat_unique_map, features_order, cts_features, cat_features, "
    + "expected_axes_count",
    [
        (
            "df_simple",
            {"cat_1": ["A", "B", "C"], "cat_2": ["X", "Y", "Z"]},
            ["cts_1", "cts_2", "cat_1", "cat_2"],
            ["cts_1", "cts_2"],
            ["cat_1", "cat_2"],
            6,
        ),
        (
            "df_simple",
            {"cat_1": ["A", "B", "C"], "cat_2": ["X", "Y", "Z"]},
            ["cat_1", "cat_2"],
            [],
            ["cat_1", "cat_2"],
            3,
        ),
        (
            "df_simple",
            {},
            ["cts_1", "cts_2"],
            ["cts_1", "cts_2"],
            [],
            3,
        ),
        (
            "df_simple",
            {},
            [],
            [],
            [],
            0,
        ),
    ],
    indirect=["df_dispatcher"],
)
def test_get_pdf_plot(
    mocker: MockerFixture,
    set_agg_backend,
    close_all_figs_after_test,
    df_dispatcher: pd.DataFrame,
    cat_unique_map: Dict[str, List[str]],
    features_order: List[str],
    cts_features: List[str],
    cat_features: List[str],
    expected_axes_count: int,
):
    spy_combiner = mocker.spy(obj=PlotCombiner, name="combine")

    result = TabularPDFPlotter.get_pdf_plot(
        dataset=df_dispatcher,
        features_order=features_order,
        cts_features=cts_features,
        cat_features=cat_features,
        cat_unique_map=cat_unique_map,
    )
    assert isinstance(result, Figure)
    spy_combiner.assert_called_once()
    assert len(spy_combiner.call_args.kwargs["plots"]) == len(features_order)
    if expected_axes_count > 0:
        assert len(result.axes) == expected_axes_count
        assert result.texts
        assert "Probability Density Functions" in result.texts[0].get_text()
    else:
        assert len(result.axes) == 0
