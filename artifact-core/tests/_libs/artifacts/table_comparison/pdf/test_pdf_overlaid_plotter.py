from typing import Dict, List, Tuple

import pandas as pd
import pytest
from artifact_core._libs.artifacts.table_comparison.pdf.overlaid_plotter import (  # noqa: E501
    TabularOverlaidPDFPlotter,
)
from artifact_core._libs.tools.plotters.overlaid_pdf_plotter import (  # noqa: E501
    OverlaidPDFPlotter,
)
from artifact_core._libs.tools.plotters.overlaid_pmf_plotter import (  # noqa: E501
    OverlaidPMFPlotter,
)
from artifact_core._libs.tools.plotters.plot_combiner import PlotCombiner
from matplotlib.figure import Figure
from pytest_mock import MockerFixture


@pytest.mark.unit
@pytest.mark.parametrize(
    "df_pair_dispatcher, cat_unique_map, features_order, cts_features, cat_features, "
    + "expected_plot_count, expected_pdf_calls, expected_pmf_calls",
    [
        (
            ("df_small_real", "df_small_synthetic"),
            {"cat_1": ["A", "B", "C"], "cat_2": ["X", "Y", "Z"]},
            ["cts_1", "cts_2", "cat_1", "cat_2"],
            ["cts_1", "cts_2"],
            ["cat_1", "cat_2"],
            4,
            2,
            2,
        ),
        (
            ("df_small_real", "df_small_synthetic"),
            {},
            ["cts_1"],
            ["cts_1"],
            [],
            1,
            1,
            0,
        ),
        (
            ("df_small_real", "df_small_synthetic"),
            {"cat_1": ["A", "B", "C"], "cat_2": ["X", "Y", "Z"]},
            ["cat_1", "cat_2"],
            [],
            ["cat_1", "cat_2"],
            2,
            0,
            2,
        ),
        (
            ("df_small_real", "df_small_synthetic"),
            {},
            [],
            [],
            [],
            0,
            0,
            0,
        ),
    ],
    indirect=["df_pair_dispatcher"],
)
def test_get_overlaid_pdf_plot_collection(
    mocker: MockerFixture,
    set_agg_backend,
    close_all_figs_after_test,
    df_pair_dispatcher: Tuple[pd.DataFrame, pd.DataFrame],
    cat_unique_map: Dict[str, List[str]],
    features_order: List[str],
    cts_features: List[str],
    cat_features: List[str],
    expected_plot_count: int,
    expected_pdf_calls: int,
    expected_pmf_calls: int,
):
    spy_pdf = mocker.spy(obj=OverlaidPDFPlotter, name="plot_overlaid_pdf")
    spy_pmf = mocker.spy(obj=OverlaidPMFPlotter, name="plot_overlaid_pmf")
    df_real, df_synthetic = df_pair_dispatcher
    result = TabularOverlaidPDFPlotter.get_overlaid_pdf_plot_collection(
        dataset_real=df_real,
        dataset_synthetic=df_synthetic,
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
        expected_title = (
            f"PMF Comparison: {feature}"
            if feature in cat_features
            else f"PDF Comparison: {feature}"
        )
        assert title_text == expected_title


@pytest.mark.unit
@pytest.mark.parametrize(
    "df_pair_dispatcher, cat_unique_map, features_order, cts_features, cat_features, "
    + "expected_axes_count",
    [
        (
            ("df_small_real", "df_small_synthetic"),
            {"cat_1": ["A", "B", "C"], "cat_2": ["X", "Y", "Z"]},
            ["cts_1", "cts_2", "cat_1", "cat_2"],
            ["cts_1", "cts_2"],
            ["cat_1", "cat_2"],
            6,
        ),
        (
            ("df_small_real", "df_small_synthetic"),
            {},
            ["cts_1"],
            ["cts_1"],
            [],
            3,
        ),
        (
            ("df_small_real", "df_small_synthetic"),
            {"cat_1": ["A", "B", "C"], "cat_2": ["X", "Y", "Z"]},
            ["cat_1", "cat_2"],
            [],
            ["cat_1", "cat_2"],
            3,
        ),
        (
            ("df_small_real", "df_small_synthetic"),
            {},
            [],
            [],
            [],
            0,
        ),
    ],
    indirect=["df_pair_dispatcher"],
)
def test_get_overlaid_pdf_plot(
    mocker: MockerFixture,
    set_agg_backend,
    close_all_figs_after_test,
    df_pair_dispatcher: Tuple[pd.DataFrame, pd.DataFrame],
    cat_unique_map: Dict[str, List[str]],
    features_order: List[str],
    cts_features: List[str],
    cat_features: List[str],
    expected_axes_count: int,
):
    spy_combiner = mocker.spy(obj=PlotCombiner, name="combine")
    df_real, df_synthetic = df_pair_dispatcher
    result = TabularOverlaidPDFPlotter.get_overlaid_pdf_plot(
        dataset_real=df_real,
        dataset_synthetic=df_synthetic,
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
        assert "Probability Density Function Comparison" in result.texts[0].get_text()
    else:
        assert len(result.axes) == 0
