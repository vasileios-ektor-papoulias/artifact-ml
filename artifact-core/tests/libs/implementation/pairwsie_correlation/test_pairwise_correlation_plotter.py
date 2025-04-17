from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from artifact_core.libs.implementation.pairwsie_correlation.calculator import (
    CategoricalAssociationType,
    ContinuousAssociationType,
)
from artifact_core.libs.implementation.pairwsie_correlation.plotter import (
    PairwiseCorrelationHeatmapPlotter,
)
from artifact_core.libs.utils.plot_combiner import PlotCombinationConfig
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
            "num1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "num2": [5.0, 4.0, 3.0, 2.0, 1.0, 0.5, 1.5, 2.5, 3.5, 4.5],
            "cat1": ["A", "B", "A", "C", "B", "D", "A", "B", "C", "D"],
            "cat2": ["X", "Y", "X", "Z", "Y", "Z", "Y", "X", "Z", "Y"],
        }
    )


@pytest.fixture
def df_synthetic() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "num1": [1.2, 2.3, 3.4, 4.5, 5.6, 6.7, 7.8, 8.9, 9.0, 10.1],
            "num2": [4.9, 3.8, 2.7, 1.6, 0.5, 0.6, 1.7, 2.8, 3.9, 4.0],
            "cat1": ["B", "A", "C", "B", "A", "C", "D", "A", "B", "C"],
            "cat2": ["Y", "X", "Z", "Y", "X", "Y", "Z", "X", "Y", "Z"],
        }
    )


@pytest.fixture
def cat_features() -> List[str]:
    return ["cat1", "cat2"]


@pytest.mark.parametrize(
    "cat_corr_type, cont_corr_type",
    [
        (CategoricalAssociationType.CRAMERS_V, ContinuousAssociationType.PEARSON),
        (CategoricalAssociationType.THEILS_U, ContinuousAssociationType.SPEARMAN),
    ],
)
def test_get_correlation_heatmap(
    set_agg_backend,
    close_all_figs_after_test,
    df_real: pd.DataFrame,
    cat_features: List[str],
    cat_corr_type: CategoricalAssociationType,
    cont_corr_type: ContinuousAssociationType,
):
    result = PairwiseCorrelationHeatmapPlotter.get_correlation_heatmap(
        categorical_correlation_type=cat_corr_type,
        continuous_correlation_type=cont_corr_type,
        dataset=df_real,
        ls_cat_features=cat_features,
    )
    assert isinstance(result, Figure), "Result should be a Figure"
    assert result.get_axes(), "Figure should have at least one axis"
    assert len(result.texts) >= 1, "Figure should have at least one text element (title)"
    title_text = result.texts[0].get_text() if result.texts else ""
    assert "Pairwise Correlations" in title_text, (
        f"Expected title to contain 'Pairwise Correlations', got '{title_text}'"
    )
    if len(result.texts) >= 2:
        subtitle = result.texts[1].get_text()
        assert cat_corr_type.name in subtitle, f"Subtitle should contain {cat_corr_type.name}"
        assert cont_corr_type.name in subtitle, f"Subtitle should contain {cont_corr_type.name}"
    ax = result.axes[0]
    assert ax.collections, "Axis should have collections (heatmap)"
    assert ax.get_xticklabels(), "Axis should have x tick labels"
    assert ax.get_yticklabels(), "Axis should have y tick labels"
    width, height = result.get_size_inches()
    assert width > 0 and height > 0, f"Figure size should be positive, got {width}x{height}"


@pytest.mark.parametrize(
    "cat_corr_type, cont_corr_type",
    [
        (CategoricalAssociationType.CRAMERS_V, ContinuousAssociationType.PEARSON),
        (CategoricalAssociationType.THEILS_U, ContinuousAssociationType.SPEARMAN),
    ],
)
def test_get_correlation_difference_heatmap(
    set_agg_backend,
    close_all_figs_after_test,
    df_real: pd.DataFrame,
    df_synthetic: pd.DataFrame,
    cat_features: List[str],
    cat_corr_type: CategoricalAssociationType,
    cont_corr_type: ContinuousAssociationType,
):
    result = PairwiseCorrelationHeatmapPlotter.get_correlation_difference_heatmap(
        categorical_correlation_type=cat_corr_type,
        continuous_correlation_type=cont_corr_type,
        dataset_real=df_real,
        dataset_synthetic=df_synthetic,
        ls_cat_features=cat_features,
    )
    assert isinstance(result, Figure), "Result should be a Figure"
    assert result.get_axes(), "Figure should have at least one axis"
    assert len(result.texts) >= 1, "Figure should have at least one text element (title)"
    title_text = result.texts[0].get_text() if result.texts else ""
    assert "Pairwise Correlation Absolute Differences" in title_text, (
        f"Expected title to contain 'Pairwise Correlation Absolute Differences', got '{title_text}'"
    )
    if len(result.texts) >= 2:
        subtitle = result.texts[1].get_text()
        assert cat_corr_type.name in subtitle, f"Subtitle should contain {cat_corr_type.name}"
        assert cont_corr_type.name in subtitle, f"Subtitle should contain {cont_corr_type.name}"
    ax = result.axes[0]
    assert ax.collections, "Axis should have collections (heatmap)"
    assert ax.get_xticklabels(), "Axis should have x tick labels"
    assert ax.get_yticklabels(), "Axis should have y tick labels"
    width, height = result.get_size_inches()
    assert width > 0 and height > 0, f"Figure size should be positive, got {width}x{height}"


@pytest.mark.parametrize(
    "cat_corr_type, cont_corr_type, expected_subplot_titles",
    [
        (
            CategoricalAssociationType.CRAMERS_V,
            ContinuousAssociationType.PEARSON,
            ["Real", "Synthetic", "Absolute Difference"],
        ),
        (
            CategoricalAssociationType.THEILS_U,
            ContinuousAssociationType.SPEARMAN,
            ["Real", "Synthetic", "Absolute Difference"],
        ),
    ],
)
def test_get_combined_correlation_plot(
    set_agg_backend,
    close_all_figs_after_test,
    df_real: pd.DataFrame,
    df_synthetic: pd.DataFrame,
    cat_features: List[str],
    cat_corr_type: CategoricalAssociationType,
    cont_corr_type: ContinuousAssociationType,
    expected_subplot_titles: List[str],
):
    result = PairwiseCorrelationHeatmapPlotter.get_combined_correlation_plot(
        categorical_correlation_type=cat_corr_type,
        continuous_correlation_type=cont_corr_type,
        dataset_real=df_real,
        dataset_synthetic=df_synthetic,
        ls_cat_features=cat_features,
    )
    assert isinstance(result, Figure), "Result should be a Figure"
    assert result.get_axes(), "Figure should have at least one axis"
    assert len(result.texts) >= 1, "Figure should have at least one text element (title)"
    title_text = result.texts[0].get_text() if result.texts else ""
    assert "Pairwise Correlation Heatmaps" in title_text, (
        f"Expected title to contain 'Pairwise Correlation Heatmaps', got '{title_text}'"
    )
    expected_axes_count = 3
    assert len(result.axes) == expected_axes_count, (
        f"Expected {expected_axes_count} axes, got {len(result.axes)}"
    )
    for i, (ax, expected_title) in enumerate(zip(result.axes, expected_subplot_titles)):
        title = ax.get_title()
        assert title is not None, f"Subplot {i} should have a title"
        assert ax.get_xticklabels(), f"Subplot {i} should have x tick labels"
        assert ax.get_yticklabels(), f"Subplot {i} should have y tick labels"
        assert ax.get_window_extent().width > 0, f"Subplot {i} width should be positive"
        assert ax.get_window_extent().height > 0, f"Subplot {i} height should be positive"


@pytest.mark.parametrize(
    "custom_config",
    [
        PlotCombinationConfig(
            n_cols=2,
            dpi=100,
            figsize_horizontal_multiplier=4,
            figsize_vertical_multiplier=4,
            combined_title="Custom Combined Title",
        ),
    ],
)
def test_get_combined_correlation_plot_with_custom_config(
    set_agg_backend,
    close_all_figs_after_test,
    df_real: pd.DataFrame,
    df_synthetic: pd.DataFrame,
    cat_features: List[str],
    custom_config: PlotCombinationConfig,
):
    result = PairwiseCorrelationHeatmapPlotter.get_combined_correlation_plot(
        categorical_correlation_type=CategoricalAssociationType.CRAMERS_V,
        continuous_correlation_type=ContinuousAssociationType.PEARSON,
        dataset_real=df_real,
        dataset_synthetic=df_synthetic,
        ls_cat_features=cat_features,
        plot_combiner_config=custom_config,
    )

    # Check basic figure properties
    assert isinstance(result, Figure), "Result should be a Figure"

    # Check title
    assert len(result.texts) >= 1, "Figure should have at least one text element (title)"
    title_text = result.texts[0].get_text() if result.texts else ""
    assert custom_config.combined_title in title_text, (
        f"Expected title to contain '{custom_config.combined_title}', got '{title_text}'"
    )

    # Check DPI
    assert result.dpi == custom_config.dpi, f"Expected DPI {custom_config.dpi}, got {result.dpi}"
