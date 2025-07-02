from typing import Dict, List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.figure import Figure, SubFigure

from artifact_core.libs.utils.autoscale.combined import (
    CombinedAutoscalerHyperparams,
    CombinedPlotAutoscaler,
)
from artifact_core.libs.utils.plot_combiner import PlotCombinationConfig, PlotCombiner


class CDFPlotter:
    _base_font_size = 16.0
    _base_title_font_size = 18.0
    _base_tick_font_size = 12.0
    _base_figure_width = 8.0
    _base_figure_height = 6.0

    _plot_color = "olive"
    _plot_marker_size = 5
    _plot_marker_edge_width = 1
    _line_width = 5
    _line_alpha = 0.5
    _plot_marker = "o"
    _line_style = "-"
    _gridline_color = "black"
    _gridline_style = ":"
    _minor_ax_grid_linewidth = 0.1
    _major_ax_grid_linewidth = 1
    _axis_font_size = "14"

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
        combined_title="Cumulative Density Functions",
        combined_title_fontsize=8,
        combined_title_vertical_position=1,
    )

    @classmethod
    def get_cdf_plot(
        cls,
        dataset: pd.DataFrame,
        ls_cts_features: List[str],
    ):
        dict_plots = cls.get_cdf_plot_collection(
            dataset=dataset,
            ls_cts_features=ls_cts_features,
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
    def get_cdf_plot_collection(
        cls,
        dataset: pd.DataFrame,
        ls_cts_features: List[str],
    ) -> Dict[str, Figure]:
        dict_plots = {}
        for feature in ls_cts_features:
            cdf_plot = cls._plot_cdf(sr_data=dataset[feature], feature_name=feature)
            dict_plots[feature] = cdf_plot
        return dict_plots

    @classmethod
    def _plot_cdf(cls, sr_data: pd.Series, feature_name: str) -> Figure:
        x = sr_data.sort_values()
        y = np.arange(1, len(sr_data) + 1) / len(sr_data)

        fig, ax = plt.subplots(figsize=(cls._base_figure_width, cls._base_figure_height))
        plt.close(fig)
        ax.set_xlabel("Values", size=cls._base_font_size)
        ax.set_ylabel("Normalized Cumulative Sum", size=cls._base_font_size)
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
        ax.plot(
            x,
            y,
            marker=cls._plot_marker,
            markersize=cls._plot_marker_size,
            markeredgewidth=cls._plot_marker_edge_width,
            linestyle=cls._line_style,
            linewidth=cls._line_width,
            alpha=cls._line_alpha,
            color=cls._plot_color,
        )
        ax.tick_params(axis="both", which="major", labelsize=cls._base_tick_font_size)
        ax.tick_params(axis="both", which="minor", labelsize=cls._base_tick_font_size * 0.8)
        plot = ax.get_figure()
        if plot is None:
            return Figure()
        if isinstance(plot, SubFigure):
            return Figure()
        plot.suptitle(f"CDF: {feature_name}", fontsize=cls._base_title_font_size)
        return plot
