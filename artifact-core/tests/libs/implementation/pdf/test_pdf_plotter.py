from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from artifact_core.libs.implementation.pdf.plotter import PDFPlotter
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
def sample_dataframe():
    return pd.DataFrame(
        {
            "continuous_feature": [1.0, 2.0, 3.0, 4.0, 5.0],
            "categorical_feature": ["A", "B", "A", "C", "B"],
            "unused_feature": ["X", "Y", "Z", "X", "Y"],
        }
    )


@pytest.fixture
def cat_unique_map():
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
def test_get_pdf_plot_collection(
    set_agg_backend,
    close_all_figs_after_test,
    sample_dataframe,
    cat_unique_map,
    features_order: List[str],
    cts_features: List[str],
    cat_features: List[str],
    expected_plot_count: int,
):
    result = PDFPlotter.get_pdf_plot_collection(
        dataset=sample_dataframe,
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
def test_get_pdf_plot(
    set_agg_backend,
    close_all_figs_after_test,
    sample_dataframe,
    cat_unique_map,
    features_order: List[str],
    cts_features: List[str],
    cat_features: List[str],
    expected_axes_count: int,
):
    result = PDFPlotter.get_pdf_plot(
        dataset=sample_dataframe,
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
            f"Expected title to contain 'Probability Density Functions', got {result.texts[0].get_text()}"
        )
    else:
        assert len(result.axes) == 0, (
            f"Expected 0 axes for empty features_order, got {len(result.axes)}"
        )


@pytest.mark.parametrize(
    "data, feature_name, expected_title",
    [
        (
            pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], name="continuous"),
            "continuous",
            "PDF: continuous",
        ),
    ],
)
def test_plot_pdf_continuous(
    set_agg_backend,
    close_all_figs_after_test,
    data: pd.Series,
    feature_name: str,
    expected_title: str,
):
    result = PDFPlotter._plot_pdf_continuous(sr_data=data, feature_name=feature_name)

    assert isinstance(result, Figure), "Result should be a Figure"
    assert result.get_axes(), "Figure should have at least one axis"
    title_text = result.texts[0].get_text() if result.texts else ""
    assert title_text == expected_title, f"Expected title '{expected_title}', got '{title_text}'"

    ax = result.get_axes()[0]
    assert ax.get_xlabel() == "Values", f"Expected x-label 'Values', got '{ax.get_xlabel()}'"
    assert ax.get_ylabel() == "Probability Density", (
        f"Expected y-label 'Probability Density', got '{ax.get_ylabel()}'"
    )
    assert len(ax.patches) > 0, "No histogram bars found in the plot"


@pytest.mark.parametrize(
    "data, feature_name, categories, expected_title",
    [
        (
            pd.Series(["A", "B", "A", "C", "B"], name="categorical"),
            "categorical",
            ["A", "B", "C"],
            "PMF: categorical",
        ),
    ],
)
def test_plot_pdf_categorical(
    set_agg_backend,
    close_all_figs_after_test,
    data: pd.Series,
    feature_name: str,
    categories: List[str],
    expected_title: str,
):
    result = PDFPlotter._plot_pdf_categorical(
        sr_data=data, feature_name=feature_name, ls_unique_categories=categories
    )

    assert isinstance(result, Figure), "Result should be a Figure"
    assert result.get_axes(), "Figure should have at least one axis"
    title_text = result.texts[0].get_text() if result.texts else ""
    assert title_text == expected_title, f"Expected title '{expected_title}', got '{title_text}'"

    ax = result.get_axes()[0]
    assert ax.get_xlabel() == feature_name, (
        f"Expected x-label '{feature_name}', got '{ax.get_xlabel()}'"
    )
    assert ax.get_ylabel() == "Probability", (
        f"Expected y-label 'Probability', got '{ax.get_ylabel()}'"
    )
    assert len(ax.patches) == len(categories), (
        f"Expected {len(categories)} bars, got {len(ax.patches)}"
    )
