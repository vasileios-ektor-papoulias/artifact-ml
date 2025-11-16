from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

from artifact_core._libs.artifacts.table_comparison.correlations.calculator import (
    CategoricalAssociationType,
    ContinuousAssociationType,
    CorrelationCalculator,
)
from artifact_core._libs.tools.plotters.plot_combiner import PlotCombinationConfig, PlotCombiner


@dataclass
class CorrelationHeatmapConfig:
    title: str
    colormap: LinearSegmentedColormap
    annotate: bool = True
    show_cbar: bool = True
    annotation_precision: str = ".2f"
    width: float = 7.0
    height: float = 7.0
    subtitle_centering: Literal["center", "left", "right"] = "center"
    layout_proportion_top: float = 0.9


class CorrelationHeatmapPlotter:
    _correlation_heatmap_config = CorrelationHeatmapConfig(
        title="Pairwise Correlations",
        colormap=sns.diverging_palette(10, 133, as_cmap=True),
    )
    _correlation_difference_heatmap_config = CorrelationHeatmapConfig(
        title="Pairwise Correlation Absolute Differences",
        colormap=sns.diverging_palette(120, 10, as_cmap=True),
    )
    _plot_combiner_config = PlotCombinationConfig(
        n_cols=3,
        dpi=150,
        figsize_horizontal_multiplier=5,
        figsize_vertical_multiplier=5,
        tight_layout_rect=(0, 0, 1, 0.95),
        tight_layout_pad=1,
        subplots_adjust_hspace=0.1,
        subplots_adjust_wspace=0.1,
        include_fig_titles=True,
        fig_title_fontsize=12,
        combined_title="Pairwise Correlation Heatmaps",
        combined_title_vertical_position=1.1,
    )
    _combined_plot_real_key = "Real"
    _combined_plot_synthetic_key = "Synthetic"
    _combined_plor_difference_key = "Absolute Difference"

    @classmethod
    def get_correlation_heatmap_collection(
        cls,
        categorical_correlation_type: CategoricalAssociationType,
        continuous_correlation_type: ContinuousAssociationType,
        dataset_real: pd.DataFrame,
        dataset_synthetic: pd.DataFrame,
        cat_features: Sequence[str],
    ) -> Mapping[str, Figure]:
        dict_plots = {}
        dict_plots[cls._combined_plot_real_key] = cls.get_correlation_heatmap(
            categorical_correlation_type=categorical_correlation_type,
            continuous_correlation_type=continuous_correlation_type,
            dataset=dataset_real,
            cat_features=cat_features,
        )
        dict_plots[cls._combined_plot_synthetic_key] = cls.get_correlation_heatmap(
            categorical_correlation_type=categorical_correlation_type,
            continuous_correlation_type=continuous_correlation_type,
            dataset=dataset_synthetic,
            cat_features=cat_features,
        )
        dict_plots[cls._combined_plor_difference_key] = cls.get_correlation_difference_heatmap(
            categorical_correlation_type=categorical_correlation_type,
            continuous_correlation_type=continuous_correlation_type,
            dataset_real=dataset_real,
            dataset_synthetic=dataset_synthetic,
            cat_features=cat_features,
        )
        return dict_plots

    @classmethod
    def get_combined_correlation_heatmaps(
        cls,
        categorical_correlation_type: CategoricalAssociationType,
        continuous_correlation_type: ContinuousAssociationType,
        dataset_real: pd.DataFrame,
        dataset_synthetic: pd.DataFrame,
        cat_features: Sequence[str],
        plot_combiner_config=_plot_combiner_config,
    ) -> Figure:
        dict_plots = {}
        dict_plots[cls._combined_plot_real_key] = cls.get_correlation_heatmap(
            categorical_correlation_type=categorical_correlation_type,
            continuous_correlation_type=continuous_correlation_type,
            dataset=dataset_real,
            cat_features=cat_features,
        )
        dict_plots[cls._combined_plot_synthetic_key] = cls.get_correlation_heatmap(
            categorical_correlation_type=categorical_correlation_type,
            continuous_correlation_type=continuous_correlation_type,
            dataset=dataset_synthetic,
            cat_features=cat_features,
        )
        dict_plots[cls._combined_plor_difference_key] = cls.get_correlation_difference_heatmap(
            categorical_correlation_type=categorical_correlation_type,
            continuous_correlation_type=continuous_correlation_type,
            dataset_real=dataset_real,
            dataset_synthetic=dataset_synthetic,
            cat_features=cat_features,
        )
        combined_plot = PlotCombiner.combine(plots=dict_plots, config=plot_combiner_config)
        return combined_plot

    @classmethod
    def get_correlation_heatmap(
        cls,
        categorical_correlation_type: CategoricalAssociationType,
        continuous_correlation_type: ContinuousAssociationType,
        dataset: pd.DataFrame,
        cat_features: Sequence[str],
    ) -> Figure:
        df_correlations = CorrelationCalculator.compute_df_correlations(
            categorical_correlation_type=categorical_correlation_type,
            continuous_correlation_type=continuous_correlation_type,
            dataset=dataset,
            cat_features=cat_features,
        )
        subtitle = cls._get_heatmap_subtitle(
            categorical_correlation_type=categorical_correlation_type,
            continuous_correlation_type=continuous_correlation_type,
        )
        fig_correlation_heatmap = cls._format_fig_heatmap(
            df_heatmap=df_correlations,
            subtitle=subtitle,
            config=cls._correlation_heatmap_config,
        )
        assert isinstance(fig_correlation_heatmap, Figure)
        return fig_correlation_heatmap

    @classmethod
    def get_correlation_difference_heatmap(
        cls,
        categorical_correlation_type: CategoricalAssociationType,
        continuous_correlation_type: ContinuousAssociationType,
        dataset_real: pd.DataFrame,
        dataset_synthetic: pd.DataFrame,
        cat_features: Sequence[str],
    ) -> Figure:
        df_correlation_difference = CorrelationCalculator.compute_df_correlation_difference(
            categorical_correlation_type=categorical_correlation_type,
            continuous_correlation_type=continuous_correlation_type,
            dataset_real=dataset_real,
            dataset_synthetic=dataset_synthetic,
            cat_features=cat_features,
        )
        subtitle = cls._get_heatmap_subtitle(
            categorical_correlation_type=categorical_correlation_type,
            continuous_correlation_type=continuous_correlation_type,
        )
        fig_correlation_difference_heatmap = cls._format_fig_heatmap(
            df_heatmap=df_correlation_difference,
            subtitle=subtitle,
            config=cls._correlation_difference_heatmap_config,
        )
        assert isinstance(fig_correlation_difference_heatmap, Figure)
        return fig_correlation_difference_heatmap

    @classmethod
    def _format_fig_heatmap(
        cls,
        df_heatmap: pd.DataFrame,
        subtitle: str,
        config: CorrelationHeatmapConfig,
    ):
        ax = sns.heatmap(
            df_heatmap,
            cmap=config.colormap,
            annot=config.annotate,
            cbar=config.show_cbar,
            fmt=config.annotation_precision,
            square=True,
            center=0,
        )
        fig_heatmap = ax.get_figure()
        assert isinstance(fig_heatmap, Figure)
        plt.close(fig_heatmap)
        fig_heatmap.set_size_inches(w=config.width, h=config.height)
        fig_heatmap.subplots_adjust(top=config.layout_proportion_top)
        fig_heatmap.suptitle(t=config.title)
        fig_heatmap.axes[0].set_title(
            label=subtitle,
            loc=config.subtitle_centering,
        )
        return fig_heatmap

    @staticmethod
    def _get_heatmap_subtitle(
        categorical_correlation_type: CategoricalAssociationType,
        continuous_correlation_type: ContinuousAssociationType,
    ) -> str:
        return (
            f"Categorical: {categorical_correlation_type.name}"
            + "|"
            + f"Continuous: {continuous_correlation_type.name}"
        )
