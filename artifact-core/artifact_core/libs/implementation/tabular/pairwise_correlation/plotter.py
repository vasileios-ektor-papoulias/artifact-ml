from dataclasses import dataclass
from typing import Dict, List

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

from artifact_core.libs.implementation.tabular.pairwise_correlation.calculator import (
    CategoricalAssociationType,
    ContinuousAssociationType,
    PairwiseCorrelationCalculator,
)
from artifact_core.libs.utils.autoscale.combined import (
    CombinedAutoscalerHyperparams,
    CombinedPlotAutoscaler,
)
from artifact_core.libs.utils.autoscale.grid import (
    GridAutoscaler,
    GridAutoscalerArgs,
    GridAutoscalerHyperparams,
)
from artifact_core.libs.utils.plot_combiner import PlotCombinationConfig, PlotCombiner


@dataclass
class CorrelationHeatmapConfig:
    title: str
    colormap: LinearSegmentedColormap
    annotate = True
    show_cbar = True
    annotation_precision = ".2f"
    subtitle_centering = "center"
    layout_proportion_top = 0.9


class PairwiseCorrelationHeatmapPlotter:
    _correlation_heatmap_config = CorrelationHeatmapConfig(
        title="Pairwise Correlations",
        colormap=sns.diverging_palette(10, 133, as_cmap=True),
    )
    _correlation_difference_heatmap_config = CorrelationHeatmapConfig(
        title="Pairwise Correlation Absolute Differences",
        colormap=sns.diverging_palette(120, 10, as_cmap=True),
    )
    _grid_scale_config = GridAutoscalerHyperparams(
        min_scale_factor=0.5,
        max_scale_factor=20.0,
        base_figure_width=5.0,
        base_figure_height=5.0,
        base_font_size=36.0,
        base_title_font_size=40.0,
        base_tick_font_size=24.0,
        base_legend_font_size=12.0,
        base_annotation_font_size=14.0,
        base_marker_size=5.0,
        base_line_width=2.0,
        text_font_scale_multiplier=5,
        title_font_scale_multiplier=7,
        tick_font_scale_multiplier=2,
        legend_font_scale_multiplier=2,
        annotation_font_scale_multiplier=2,
        grid_cells_per_base_size=0.5,
    )
    _combination_scale_config = CombinedAutoscalerHyperparams(
        min_scale_factor=0.5,
        max_scale_factor=10,
    )
    _plot_combiner_config = PlotCombinationConfig(
        n_cols=3,
        dpi=100,
        figsize_horizontal_multiplier=15,
        figsize_vertical_multiplier=12,
        tight_layout_rect=(0, 0, 1, 0.95),
        tight_layout_pad=0.1,
        subplots_adjust_hspace=0.05,
        subplots_adjust_wspace=0.05,
        include_fig_titles=True,
        fig_title_fontsize=4,
        combined_title="Pairwise Correlation Heatmaps",
        combined_title_fontsize=5,
        combined_title_vertical_position=1.1,
    )
    _combined_plot_real_key = "Real"
    _combined_plot_synthetic_key = "Synthetic"
    _combined_plor_difference_key = "Absolute Difference"

    @classmethod
    def get_combined_correlation_plot(
        cls,
        categorical_correlation_type: CategoricalAssociationType,
        continuous_correlation_type: ContinuousAssociationType,
        dataset_real: pd.DataFrame,
        dataset_synthetic: pd.DataFrame,
        ls_cat_features: List[str],
    ) -> Figure:
        dict_plots: Dict[str, Figure] = {}
        dict_plots[cls._combined_plot_real_key] = cls.get_correlation_heatmap(
            categorical_correlation_type=categorical_correlation_type,
            continuous_correlation_type=continuous_correlation_type,
            dataset=dataset_real,
            ls_cat_features=ls_cat_features,
        )
        dict_plots[cls._combined_plot_synthetic_key] = cls.get_correlation_heatmap(
            categorical_correlation_type=categorical_correlation_type,
            continuous_correlation_type=continuous_correlation_type,
            dataset=dataset_synthetic,
            ls_cat_features=ls_cat_features,
        )
        dict_plots[cls._combined_plor_difference_key] = cls.get_correlation_difference_heatmap(
            categorical_correlation_type=categorical_correlation_type,
            continuous_correlation_type=continuous_correlation_type,
            dataset_real=dataset_real,
            dataset_synthetic=dataset_synthetic,
            ls_cat_features=ls_cat_features,
        )
        autoscaled_config = CombinedPlotAutoscaler.get_scaled_combiner_config(
            base_config=cls._plot_combiner_config,
            num_plots=len(dict_plots),
            ls_subplot_dims=[plot.get_size_inches() for plot in dict_plots.values()],
            scale_config=cls._combination_scale_config,
        )
        combined_plot = PlotCombiner.combine(dict_plots=dict_plots, config=autoscaled_config)
        return combined_plot

    @classmethod
    def get_correlation_plot_collection(
        cls,
        categorical_correlation_type: CategoricalAssociationType,
        continuous_correlation_type: ContinuousAssociationType,
        dataset_real: pd.DataFrame,
        dataset_synthetic: pd.DataFrame,
        ls_cat_features: List[str],
    ) -> Dict[str, Figure]:
        dict_plots = {}
        dict_plots[cls._combined_plot_real_key] = cls.get_correlation_heatmap(
            categorical_correlation_type=categorical_correlation_type,
            continuous_correlation_type=continuous_correlation_type,
            dataset=dataset_real,
            ls_cat_features=ls_cat_features,
        )
        dict_plots[cls._combined_plot_synthetic_key] = cls.get_correlation_heatmap(
            categorical_correlation_type=categorical_correlation_type,
            continuous_correlation_type=continuous_correlation_type,
            dataset=dataset_synthetic,
            ls_cat_features=ls_cat_features,
        )
        dict_plots[cls._combined_plor_difference_key] = cls.get_correlation_difference_heatmap(
            categorical_correlation_type=categorical_correlation_type,
            continuous_correlation_type=continuous_correlation_type,
            dataset_real=dataset_real,
            dataset_synthetic=dataset_synthetic,
            ls_cat_features=ls_cat_features,
        )
        return dict_plots

    @classmethod
    def get_correlation_heatmap(
        cls,
        categorical_correlation_type: CategoricalAssociationType,
        continuous_correlation_type: ContinuousAssociationType,
        dataset: pd.DataFrame,
        ls_cat_features: List[str],
    ) -> Figure:
        df_correlations = PairwiseCorrelationCalculator.compute_df_correlations(
            categorical_correlation_type=categorical_correlation_type,
            continuous_correlation_type=continuous_correlation_type,
            dataset=dataset,
            ls_cat_features=ls_cat_features,
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
        return fig_correlation_heatmap

    @classmethod
    def get_correlation_difference_heatmap(
        cls,
        categorical_correlation_type: CategoricalAssociationType,
        continuous_correlation_type: ContinuousAssociationType,
        dataset_real: pd.DataFrame,
        dataset_synthetic: pd.DataFrame,
        ls_cat_features: List[str],
    ) -> Figure:
        df_correlation_difference = PairwiseCorrelationCalculator.compute_df_correlation_difference(
            categorical_correlation_type=categorical_correlation_type,
            continuous_correlation_type=continuous_correlation_type,
            dataset_real=dataset_real,
            dataset_synthetic=dataset_synthetic,
            ls_cat_features=ls_cat_features,
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
        return fig_correlation_difference_heatmap

    @classmethod
    def _format_fig_heatmap(
        cls,
        df_heatmap: pd.DataFrame,
        subtitle: str,
        config: CorrelationHeatmapConfig,
    ):
        grid_args = GridAutoscalerArgs(
            grid_width=df_heatmap.shape[1], grid_height=df_heatmap.shape[0]
        )
        scale = GridAutoscaler.compute(grid_args, cls._grid_scale_config)
        fig_width = scale.figure_width
        fig_height = scale.figure_height
        ax = sns.heatmap(
            df_heatmap,
            cmap=config.colormap,
            annot=config.annotate,
            cbar=config.show_cbar,
            fmt=config.annotation_precision,
            square=True,
            center=0,
            annot_kws={"fontsize": scale.annotation_font_size},
        )
        fig_heatmap = ax.get_figure()
        plt.close(fig_heatmap)
        fig_heatmap.set_size_inches(w=fig_width, h=fig_height)
        fig_heatmap.subplots_adjust(top=config.layout_proportion_top)
        fig_heatmap.suptitle(t=config.title, fontsize=scale.title_font_size)
        fig_heatmap.axes[0].set_title(
            label=subtitle,
            loc=config.subtitle_centering,
            fontsize=scale.font_size,
        )
        ax.tick_params(axis="both", which="major", labelsize=scale.tick_font_size)
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
