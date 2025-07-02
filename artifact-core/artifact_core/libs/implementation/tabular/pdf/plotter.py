from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from artifact_core.libs.utils.autoscale.categorical import (
    CategoricalAutoscaler,
    CategoricalAutoscalerArgs,
    CategoricalAutoscalerHyperparams,
)
from artifact_core.libs.utils.autoscale.combined import (
    CombinedAutoscalerHyperparams,
    CombinedPlotAutoscaler,
)
from artifact_core.libs.utils.plot_combiner import PlotCombinationConfig, PlotCombiner


class PDFPlotter:
    # Private base sizing attributes for continuous plots
    _base_font_size = 16.0
    _base_title_font_size = 18.0
    _base_tick_font_size = 12.0
    _base_figure_width = 8.0
    _base_figure_height = 6.0

    _plot_color = "olive"
    _gridline_color = "black"
    _gridline_style = ":"
    _axis_font_size = "14"
    _minor_ax_grid_linewidth = 0.1
    _major_ax_grid_linewidth = 1
    _cts_density_n_bins = 50
    _cts_density_enable_kde = True
    _cts_densitiy_alpha = 0.7

    _cat_autoscaler_hyperparams = CategoricalAutoscalerHyperparams(
        min_scale_factor=0.5,
        max_scale_factor=100.0,
        categories_per_base_width=5,
        base_figure_width=4.0,
        base_figure_height=7.0,
        base_font_size=14.0,
        base_title_font_size=16.0,
        base_tick_font_size=12.0,
        base_legend_font_size=12.0,
        base_marker_size=5.0,
        base_line_width=2.0,
    )
    _combination_scale_config = CombinedAutoscalerHyperparams(
        min_scale_factor=0.5,
        max_scale_factor=10.0,
    )
    _plot_combiner_config = PlotCombinationConfig(
        n_cols=3,
        dpi=150,
        figsize_horizontal_multiplier=6,
        figsize_vertical_multiplier=4,
        tight_layout_rect=(0, 0, 1, 0.95),
        tight_layout_pad=0.1,
        subplots_adjust_hspace=0.1,
        subplots_adjust_wspace=0.1,
        fig_title_fontsize=5,
        include_fig_titles=False,
        combined_title="Probability Density Functions",
        combined_title_fontsize=8,
        combined_title_vertical_position=1,
    )

    @classmethod
    def get_pdf_plot(
        cls,
        dataset: pd.DataFrame,
        ls_features_order: List[str],
        ls_cts_features: List[str],
        ls_cat_features: List[str],
        cat_unique_map: Dict[str, List[str]],
    ) -> Figure:
        dict_plots = cls.get_pdf_plot_collection(
            dataset=dataset,
            ls_features_order=ls_features_order,
            ls_cts_features=ls_cts_features,
            ls_cat_features=ls_cat_features,
            cat_unique_map=cat_unique_map,
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
    def get_pdf_plot_collection(
        cls,
        dataset: pd.DataFrame,
        ls_features_order: List[str],
        ls_cts_features: List[str],
        ls_cat_features: List[str],
        cat_unique_map: Dict[str, List[str]],
    ) -> Dict[str, Figure]:
        dict_plots = {}
        for feature in ls_features_order:
            if feature in ls_cat_features:
                fig = cls._plot_pdf_categorical(
                    sr_data=dataset[feature],
                    feature_name=feature,
                    ls_unique_categories=cat_unique_map.get(feature, []),
                )
            elif feature in ls_cts_features:
                fig = cls._plot_pdf_continuous(
                    sr_data=dataset[feature],
                    feature_name=feature,
                )
            else:
                continue
            dict_plots[feature] = fig
        return dict_plots

    @classmethod
    def _plot_pdf_categorical(
        cls,
        sr_data: pd.Series,
        feature_name: str,
        ls_unique_categories: List[str],
    ) -> Figure:
        freq_series = sr_data.value_counts(normalize=True).reindex(
            ls_unique_categories, fill_value=0
        )

        # Get autoscaled figure size and font sizes
        num_categories = len(ls_unique_categories)
        categorical_args = CategoricalAutoscalerArgs(num_categories=num_categories)
        scale_result = CategoricalAutoscaler.compute(
            args=categorical_args, params=cls._cat_autoscaler_hyperparams
        )
        fig_width = scale_result.figure_width
        fig_height = scale_result.figure_height

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        plt.close(fig)
        positions = range(len(freq_series.index))
        ax.bar(positions, freq_series.values, color=cls._plot_color, alpha=0.7)
        ax.set_xticks(positions)
        ax.set_xticklabels(
            freq_series.index.astype(str), rotation="vertical", fontsize=scale_result.tick_font_size
        )

        ax.set_xlabel(feature_name, fontsize=scale_result.font_size)
        ax.set_ylabel("Probability", fontsize=scale_result.font_size)
        fig.suptitle(f"PMF: {feature_name}", fontsize=scale_result.title_font_size)
        return fig

    @classmethod
    def _plot_pdf_continuous(cls, sr_data: pd.Series, feature_name: str) -> Figure:
        # Use base configuration for consistent sizing
        fig_width = cls._base_figure_width
        fig_height = cls._base_figure_height

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        plt.close(fig)
        ax.set_xlabel("Values", fontsize=cls._base_font_size)
        ax.set_ylabel("Probability Density", fontsize=cls._base_font_size)
        ax.set_axisbelow(True)
        ax.grid(
            True,
            which="minor",
            linestyle=cls._gridline_style,
            linewidth=cls._minor_ax_grid_linewidth,
            color=cls._gridline_color,
        )
        ax.grid(
            True,
            which="major",
            linestyle=cls._gridline_style,
            linewidth=cls._major_ax_grid_linewidth,
            color=cls._gridline_color,
        )
        sns.histplot(
            sr_data.dropna(),
            bins=cls._cts_density_n_bins,
            stat="density",
            color=cls._plot_color,
            alpha=cls._cts_densitiy_alpha,
            kde=cls._cts_density_enable_kde,
            ax=ax,
        )

        # Set tick label font sizes
        ax.tick_params(axis="both", which="major", labelsize=cls._base_tick_font_size)
        ax.tick_params(axis="both", which="minor", labelsize=cls._base_tick_font_size * 0.8)

        fig.suptitle(f"PDF: {feature_name}", fontsize=cls._base_title_font_size)
        return fig
