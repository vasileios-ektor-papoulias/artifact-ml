from typing import Dict, List, Tuple

import pandas as pd
import pytest
from artifact_core._libs.implementation.tabular.pdf.overlaid_plotter import (
    TabularOverlaidPDFPlotter,
)
from matplotlib.figure import Figure


@pytest.mark.unit
@pytest.mark.parametrize(
    "df_pair_dispatcher, cat_unique_map, features_order, cts_features, cat_features, "
    + "expected_plot_count",
    [
        (
            ("df_small_real", "df_small_synthetic"),
            {"cat_1": ["A", "B", "C"], "cat_2": ["X", "Y", "Z"]},
            ["cts_1", "cts_2", "cat_1", "cat_2"],
            [
                "cts_1",
                "cts_2",
            ],
            ["cat_1", "cat_2"],
            4,
        ),
        (
            ("df_small_real", "df_small_synthetic"),
            {},
            ["cts_1"],
            [
                "cts_1",
            ],
            [],
            1,
        ),
        (
            ("df_small_real", "df_small_synthetic"),
            {"cat_1": ["A", "B", "C"], "cat_2": ["X", "Y", "Z"]},
            ["cat_1", "cat_2"],
            [],
            ["cat_1", "cat_2"],
            2,
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
def test_get_overlaid_pdf_plot_collection(
    set_agg_backend,
    close_all_figs_after_test,
    df_pair_dispatcher: Tuple[pd.DataFrame, pd.DataFrame],
    cat_unique_map: Dict[str, List[str]],
    features_order: List[str],
    cts_features: List[str],
    cat_features: List[str],
    expected_plot_count: int,
):
    df_real, df_synthetic = df_pair_dispatcher
    result = TabularOverlaidPDFPlotter.get_overlaid_pdf_plot_collection(
        dataset_real=df_real,
        dataset_synthetic=df_synthetic,
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
        expected_title = (
            f"PMF Comparison: {feature}"
            if feature in cat_features
            else f"PDF Comparison: {feature}"
        )
        assert title_text == expected_title, (
            f"Expected title '{expected_title}', got '{title_text}'"
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "df_pair_dispatcher, cat_unique_map, features_order, cts_features, cat_features, "
    + "expected_axes_count",
    [
        (
            ("df_small_real", "df_small_synthetic"),
            {"cat_1": ["A", "B", "C"], "cat_2": ["X", "Y", "Z"]},
            ["cts_1", "cts_2", "cat_1", "cat_2"],
            [
                "cts_1",
                "cts_2",
            ],
            ["cat_1", "cat_2"],
            6,
        ),
        (
            ("df_small_real", "df_small_synthetic"),
            {},
            ["cts_1"],
            [
                "cts_1",
            ],
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
    set_agg_backend,
    close_all_figs_after_test,
    df_pair_dispatcher: Tuple[pd.DataFrame, pd.DataFrame],
    cat_unique_map: Dict[str, List[str]],
    features_order: List[str],
    cts_features: List[str],
    cat_features: List[str],
    expected_axes_count: int,
):
    df_real, df_synthetic = df_pair_dispatcher
    result = TabularOverlaidPDFPlotter.get_overlaid_pdf_plot(
        dataset_real=df_real,
        dataset_synthetic=df_synthetic,
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
        assert "Probability Density Function Comparison" in result.texts[0].get_text(), (
            "Expected title to contain 'Probability Density Function Comparison', "
            + f"got {result.texts[0].get_text()}"
        )
    else:
        assert len(result.axes) == 0, (
            f"Expected 0 axes for empty ls_features_order, got {len(result.axes)}"
        )
