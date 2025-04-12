from typing import Dict, List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from artifact_core.libs.utils.plot_combiner import (
    PlotCombinationConfig,
    PlotCombiner,
)


class CDFPlotter:
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
        combined_plot = PlotCombiner.combine(
            dict_plots=dict_plots, config=cls._plot_combiner_config
        )
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
        fig, ax = plt.subplots()
        plt.close(fig)
        ax.set_xlabel("Values", size=cls._axis_font_size)
        ax.set_ylabel("Normalized Cumulative Sum", size=cls._axis_font_size)
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
        ax.tick_params(axis="both", which="major", labelsize=8)
        ax.tick_params(axis="both", which="minor", labelsize=6)
        plot = ax.get_figure()
        plot.suptitle(f"CDF: {feature_name}")
        return plot
