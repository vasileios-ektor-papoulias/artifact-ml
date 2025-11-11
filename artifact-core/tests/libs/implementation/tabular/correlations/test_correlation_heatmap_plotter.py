from typing import List, Tuple

import pandas as pd
import pytest
from artifact_core._libs.implementation.tabular.correlations.calculator import (
    CategoricalAssociationType,
    ContinuousAssociationType,
)
from artifact_core._libs.implementation.tabular.correlations.heatmap_plotter import (
    CorrelationHeatmapPlotter,
)
from artifact_core._libs.utils.plotters.plot_combiner import PlotCombinationConfig
from matplotlib.figure import Figure


@pytest.mark.unit
@pytest.mark.parametrize(
    "df_dispatcher, cat_features, cat_corr_type, cont_corr_type",
    [
        (
            "df_simple",
            ["cat_1", "cat_2"],
            CategoricalAssociationType.CRAMERS_V,
            ContinuousAssociationType.PEARSON,
        ),
        (
            "df_simple",
            ["cat_1", "cat_2"],
            CategoricalAssociationType.THEILS_U,
            ContinuousAssociationType.SPEARMAN,
        ),
    ],
    indirect=["df_dispatcher"],
)
def test_get_correlation_heatmap(
    set_agg_backend,
    close_all_figs_after_test,
    df_dispatcher: pd.DataFrame,
    cat_features: List[str],
    cat_corr_type: CategoricalAssociationType,
    cont_corr_type: ContinuousAssociationType,
):
    df = df_dispatcher
    result = CorrelationHeatmapPlotter.get_correlation_heatmap(
        categorical_correlation_type=cat_corr_type,
        continuous_correlation_type=cont_corr_type,
        dataset=df,
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


@pytest.mark.unit
@pytest.mark.parametrize(
    "df_pair_dispatcher, cat_features, cat_corr_type, cont_corr_type",
    [
        (
            ("df_small_real", "df_small_synthetic"),
            ["cat_1", "cat_2"],
            CategoricalAssociationType.CRAMERS_V,
            ContinuousAssociationType.PEARSON,
        ),
        (
            ("df_small_real", "df_small_synthetic"),
            ["cat_1", "cat_2"],
            CategoricalAssociationType.THEILS_U,
            ContinuousAssociationType.SPEARMAN,
        ),
    ],
    indirect=["df_pair_dispatcher"],
)
def test_get_correlation_difference_heatmap(
    set_agg_backend,
    close_all_figs_after_test,
    df_pair_dispatcher: Tuple[pd.DataFrame, pd.DataFrame],
    cat_features: List[str],
    cat_corr_type: CategoricalAssociationType,
    cont_corr_type: ContinuousAssociationType,
):
    df_real, df_synthetic = df_pair_dispatcher
    result = CorrelationHeatmapPlotter.get_correlation_difference_heatmap(
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


@pytest.mark.unit
@pytest.mark.parametrize(
    "df_pair_dispatcher, cat_features, cat_corr_type, cont_corr_type, expected_subplot_titles",
    [
        (
            ("df_small_real", "df_small_synthetic"),
            ["cat_1", "cat_2"],
            CategoricalAssociationType.CRAMERS_V,
            ContinuousAssociationType.PEARSON,
            ["Real", "Synthetic", "Absolute Difference"],
        ),
        (
            ("df_small_real", "df_small_synthetic"),
            ["cat_1", "cat_2"],
            CategoricalAssociationType.THEILS_U,
            ContinuousAssociationType.SPEARMAN,
            ["Real", "Synthetic", "Absolute Difference"],
        ),
    ],
    indirect=["df_pair_dispatcher"],
)
def test_get_combined_correlation_plot(
    set_agg_backend,
    close_all_figs_after_test,
    df_pair_dispatcher: Tuple[pd.DataFrame, pd.DataFrame],
    cat_features: List[str],
    cat_corr_type: CategoricalAssociationType,
    cont_corr_type: ContinuousAssociationType,
    expected_subplot_titles: List[str],
):
    df_real, df_synthetic = df_pair_dispatcher
    result = CorrelationHeatmapPlotter.get_combined_correlation_heatmaps(
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


@pytest.mark.unit
@pytest.mark.parametrize(
    "df_pair_dispatcher, cat_features, custom_config",
    [
        (
            ("df_small_real", "df_small_synthetic"),
            ["cat_1", "cat_2"],
            PlotCombinationConfig(
                n_cols=2,
                dpi=100,
                figsize_horizontal_multiplier=4,
                figsize_vertical_multiplier=4,
                combined_title="Custom Combined Title",
            ),
        ),
    ],
    indirect=["df_pair_dispatcher"],
)
def test_get_combined_correlation_plot_with_custom_config(
    set_agg_backend,
    close_all_figs_after_test,
    df_pair_dispatcher: Tuple[pd.DataFrame, pd.DataFrame],
    cat_features: List[str],
    custom_config: PlotCombinationConfig,
):
    df_real, df_synthetic = df_pair_dispatcher
    result = CorrelationHeatmapPlotter.get_combined_correlation_heatmaps(
        categorical_correlation_type=CategoricalAssociationType.CRAMERS_V,
        continuous_correlation_type=ContinuousAssociationType.PEARSON,
        dataset_real=df_real,
        dataset_synthetic=df_synthetic,
        ls_cat_features=cat_features,
        plot_combiner_config=custom_config,
    )

    assert isinstance(result, Figure), "Result should be a Figure"
    assert len(result.texts) >= 1, "Figure should have at least one text element (title)"
    title_text = result.texts[0].get_text() if result.texts else ""
    expected_title = (
        custom_config.combined_title if custom_config.combined_title is not None else ""
    )
    assert expected_title in title_text, (
        f"Expected title to contain '{custom_config.combined_title}', got '{title_text}'"
    )
    assert result.dpi == custom_config.dpi, f"Expected DPI {custom_config.dpi}, got {result.dpi}"
