from typing import Dict, List

import pandas as pd
import pytest
from artifact_core._libs.artifacts.table_comparison.pdf.plotter import TabularPDFPlotter


@pytest.mark.unit
@pytest.mark.parametrize(
    "df_dispatcher, cat_unique_map, features_order, cts_features, cat_features, "
    + "expected_plot_count",
    [
        (
            "df_simple",
            {"cat_1": ["A, B, C"], "cat_2": ["X", "Y", "Z"]},
            ["cts_1", "cts_2", "cat_1", "cat_2"],
            ["cts_1", "cts_2"],
            ["cat_1", "cat_2"],
            4,
        ),
        (
            "df_simple",
            {"cat_1": ["A, B, C"], "cat_2": ["X", "Y", "Z"]},
            ["cat_1", "cat_2"],
            [],
            ["cat_1", "cat_2"],
            2,
        ),
        (
            "df_simple",
            {},
            ["cts_1", "cts_2"],
            ["cts_1", "cts_2"],
            [],
            2,
        ),
        (
            "df_simple",
            {},
            [],
            [],
            [],
            0,
        ),
        (
            "df_complex",
            {
                "cat_1": ["A, B, C"],
                "cat_2": [True, False],
                "cat_3": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"],
                "cat_4": ["X", "Y", "Z"],
            },
            ["cts_1", "cts_2", "cat_1", "cat_2", "cat_3", "cat_4"],
            ["cts_1", "cts_2"],
            ["cat_1", "cat_1", "cat_2", "cat_3", "cat_4"],
            6,
        ),
        (
            "df_complex",
            {
                "cat_1": ["A, B, C"],
                "cat_2": [True, False],
                "cat_3": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"],
                "cat_4": ["X", "Y", "Z"],
            },
            ["cat_1", "cat_2", "cat_3", "cat_4"],
            [],
            ["cat_1", "cat_1", "cat_2", "cat_3", "cat_4"],
            4,
        ),
        (
            "df_complex",
            {},
            ["cts_1", "cts_2"],
            [],
            ["cts_1", "cts_2"],
            2,
        ),
        (
            "df_complex",
            {},
            [],
            [],
            [],
            0,
        ),
    ],
    indirect=["df_dispatcher"],
)
def test_get_pdf_plot_collection(
    set_agg_backend,
    close_all_figs_after_test,
    df_dispatcher: pd.DataFrame,
    cat_unique_map: Dict[str, List[str]],
    features_order: List[str],
    cts_features: List[str],
    cat_features: List[str],
    expected_plot_count: int,
):
    df = df_dispatcher
    result = TabularPDFPlotter.get_pdf_plot_collection(
        dataset=df,
        ls_features_order=features_order,
        ls_cts_features=cts_features,
        ls_cat_features=cat_features,
        cat_unique_map=cat_unique_map,
    )

    assert isinstance(result, dict), "Result should be a dictionary"
    assert len(result) == expected_plot_count, (
        f"Expected {expected_plot_count} plots, got {len(result)}"
    )

    for feature, fig in result.items():
        assert isinstance(fig, Figure), f"Expected Figure for feature {feature}, got {type(fig)}"
        assert fig.get_axes(), f"Figure for feature {feature} should have at least one axis"
        title_text = fig.texts[0].get_text() if fig.texts else ""
        assert title_text == f"PMF: {feature}" or title_text == f"PDF: {feature}", (
            f"Figure title should be 'PMF: {feature}' or 'PDF: {feature}', got {title_text}"
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "df_dispatcher, cat_unique_map, features_order, cts_features, cat_features, "
    + "expected_axes_count",
    [
        (
            "df_simple",
            {"cat_1": ["A, B, C"], "cat_2": ["X", "Y", "Z"]},
            ["cts_1", "cts_2", "cat_1", "cat_2"],
            ["cts_1", "cts_2"],
            ["cat_1", "cat_2"],
            6,
        ),
        (
            "df_simple",
            {"cat_1": ["A, B, C"], "cat_2": ["X", "Y", "Z"]},
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
        (
            "df_complex",
            {
                "cat_1": ["A, B, C"],
                "cat_2": [True, False],
                "cat_3": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"],
                "cat_4": ["X", "Y", "Z"],
            },
            ["cts_1", "cts_2", "cat_1", "cat_2", "cat_3", "cat_4"],
            ["cts_1", "cts_2"],
            ["cat_1", "cat_1", "cat_2", "cat_3", "cat_4"],
            6,
        ),
        (
            "df_complex",
            {
                "cat_1": ["A, B, C"],
                "cat_2": [True, False],
                "cat_3": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"],
                "cat_4": ["X", "Y", "Z"],
            },
            ["cat_1", "cat_2", "cat_3", "cat_4"],
            [],
            ["cat_1", "cat_1", "cat_2", "cat_3", "cat_4"],
            6,
        ),
        (
            "df_complex",
            {},
            ["cts_1", "cts_2"],
            [],
            ["cts_1", "cts_2"],
            3,
        ),
        (
            "df_complex",
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
    set_agg_backend,
    close_all_figs_after_test,
    df_dispatcher: pd.DataFrame,
    cat_unique_map: Dict[str, List[str]],
    features_order: List[str],
    cts_features: List[str],
    cat_features: List[str],
    expected_axes_count: int,
):
    df = df_dispatcher
    result = TabularPDFPlotter.get_pdf_plot(
        dataset=df,
        ls_features_order=features_order,
        ls_cts_features=cts_features,
        ls_cat_features=cat_features,
        cat_unique_map=cat_unique_map,
    )

    assert isinstance(result, Figure), "Result should be a Figure"
    if expected_axes_count > 0:
        assert len(result.axes) == expected_axes_count, (
            f"Expected {expected_axes_count} axes, got {len(result.axes)}"
        )
        assert result.texts, "Figure should have title text"
        assert "Probability Density Functions" in result.texts[0].get_text(), (
            "Expected title to contain 'Probability Density Functions', "
            + "got {result.texts[0].get_text()}"
        )
    else:
        assert len(result.axes) == 0, (
            f"Expected 0 axes for empty features_order, got {len(result.axes)}"
        )
