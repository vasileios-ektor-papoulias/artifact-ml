from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from artifact_core.libs.implementation.pdf.overlaid_plotter import OverlaidPDFPlotter
from matplotlib.figure import Figure


@pytest.fixture
def set_agg_backend():
    import matplotlib

    matplotlib.use("Agg")


@pytest.fixture
def close_all_figs_after_test():
    yield
    plt.close("all")


@pytest.fixture
def df_real() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "continuous_feature": [1.0, 2.0, 3.0, 4.0, 5.0],
            "categorical_feature": ["A", "B", "A", "C", "B"],
            "unused_feature": ["X", "Y", "Z", "X", "Y"],
        }
    )


@pytest.fixture
def df_synthetic() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "continuous_feature": [1.5, 2.5, 3.5, 4.5, 5.5],
            "categorical_feature": ["B", "A", "C", "B", "A"],
            "unused_feature": ["Y", "X", "Y", "Z", "X"],
        }
    )


@pytest.fixture
def cat_unique_map() -> Dict[str, List[str]]:
    return {"categorical_feature": ["A", "B", "C"]}


@pytest.mark.parametrize(
    "features_order, cts_features, cat_features, expected_plot_count",
    [
        (
            ["continuous_feature", "categorical_feature"],
            ["continuous_feature"],
            ["categorical_feature"],
            2,
        ),
        (
            ["continuous_feature"],
            ["continuous_feature"],
            [],
            1,
        ),
        (
            ["categorical_feature"],
            [],
            ["categorical_feature"],
            1,
        ),
        (
            ["continuous_feature", "categorical_feature", "unused_feature"],
            ["continuous_feature"],
            ["categorical_feature"],
            2,
        ),
        (
            [],
            ["continuous_feature"],
            ["categorical_feature"],
            0,
        ),
    ],
)
def test_get_overlaid_pdf_plot_collection(
    set_agg_backend,
    close_all_figs_after_test,
    df_real: pd.DataFrame,
    df_synthetic: pd.DataFrame,
    cat_unique_map: Dict[str, List[str]],
    features_order: List[str],
    cts_features: List[str],
    cat_features: List[str],
    expected_plot_count: int,
):
    result = OverlaidPDFPlotter.get_overlaid_pdf_plot_collection(
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


@pytest.mark.parametrize(
    "features_order, cts_features, cat_features, expected_axes_count",
    [
        (
            ["continuous_feature", "categorical_feature"],
            ["continuous_feature"],
            ["categorical_feature"],
            3,  # 2 features with 3 cols = 3 axes (1 will be empty)
        ),
        (
            ["continuous_feature"],
            ["continuous_feature"],
            [],
            3,  # 1 feature with 3 cols = 3 axes (2 will be empty)
        ),
        (
            ["categorical_feature"],
            [],
            ["categorical_feature"],
            3,  # 1 feature with 3 cols = 3 axes (2 will be empty)
        ),
        (
            [],
            ["continuous_feature"],
            ["categorical_feature"],
            0,  # No features in order = no axes
        ),
    ],
)
def test_get_overlaid_pdf_plot(
    set_agg_backend,
    close_all_figs_after_test,
    df_real: pd.DataFrame,
    df_synthetic: pd.DataFrame,
    cat_unique_map: Dict[str, List[str]],
    features_order: List[str],
    cts_features: List[str],
    cat_features: List[str],
    expected_axes_count: int,
):
    result = OverlaidPDFPlotter.get_overlaid_pdf_plot(
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
            f"Expected title to contain 'Probability Density Function Comparison', got {result.texts[0].get_text()}"
        )
    else:
        assert len(result.axes) == 0, (
            f"Expected 0 axes for empty features_order, got {len(result.axes)}"
        )
