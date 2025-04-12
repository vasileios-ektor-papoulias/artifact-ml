from typing import Dict, List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from artifact_core.libs.utils.plot_combiner import (
    PlotCombinationConfig,
    PlotCombiner,
)


class OverlaidCDFPlotter:
    _label_real = "Real"
    _plot_color_real = "olive"
    _plot_marker_size_real = 5
    _plot_marker_edge_width_real = 1
    _line_width_real = 5
    _line_alpha_real = 0.5

    _label_sythetic = "Synthetic"
    _plot_color_synthetic = "crimson"
    _plot_marker_size_synthetic = 5
    _plot_marker_edge_width_synthetic = 1
    _line_width_synthetic = 5
    _line_alpha_synthetic = 0.5

    _line_style = "-"
    _plot_marker = "o"
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
        combined_title="Cumulative Density Function Comparison",
        combined_title_vertical_position=1,
    )

    @classmethod
    def get_overlaid_cdf_plot(
        cls,
        dataset_real: pd.DataFrame,
        dataset_synthetic: pd.DataFrame,
        ls_cts_features: List[str],
    ) -> Figure:
        dict_plots = cls.get_overlaid_cdf_plot_collection(
            dataset_real=dataset_real,
            dataset_synthetic=dataset_synthetic,
            ls_cts_features=ls_cts_features,
        )
        combined_plot = PlotCombiner.combine(
            dict_plots=dict_plots, config=cls._plot_combiner_config
        )
        return combined_plot

    @classmethod
    def get_overlaid_cdf_plot_collection(
        cls,
        dataset_real: pd.DataFrame,
        dataset_synthetic: pd.DataFrame,
        ls_cts_features: List[str],
    ) -> Dict[str, Figure]:
        dict_plots = {}
        for feature in ls_cts_features:
            cdf_plot = cls._plot_overlaid_cdf(
                sr_data_real=dataset_real[feature],
                sr_data_synthetic=dataset_synthetic[feature],
                feature_name=feature,
            )
            dict_plots[feature] = cdf_plot
        return dict_plots

    @classmethod
    def _plot_overlaid_cdf(
        cls,
        sr_data_real: pd.Series,
        sr_data_synthetic: pd.Series,
        feature_name: str,
    ) -> Figure:
        x1 = sr_data_real.sort_values()
        y1 = np.arange(1, len(sr_data_real) + 1) / len(sr_data_real)
        x2 = sr_data_synthetic.sort_values()
        y2 = np.arange(1, len(sr_data_synthetic) + 1) / len(sr_data_synthetic)
        fig, ax = plt.subplots()
        plt.close(fig)
        ax.set_xlabel("Values", size=cls._axis_font_size)
        ax.set_ylabel("Normalized Cumulative Sum", size=cls._axis_font_size)
        ax.set_axisbelow(True)
        ax.minorticks_on()
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
            x1,
            y1,
            marker=cls._plot_marker,
            markersize=cls._plot_marker_size_real,
            markeredgewidth=cls._plot_marker_edge_width_real,
            linestyle=cls._line_style,
            linewidth=cls._line_width_real,
            alpha=cls._line_alpha_real,
            color=cls._plot_color_real,
            label=cls._label_real,
        )
        ax.plot(
            x2,
            y2,
            marker=cls._plot_marker,
            markersize=cls._plot_marker_size_synthetic,
            markeredgewidth=cls._plot_marker_edge_width_synthetic,
            linestyle=cls._line_style,
            linewidth=cls._line_width_synthetic,
            alpha=cls._line_alpha_synthetic,
            color=cls._plot_color_synthetic,
            label=cls._label_sythetic,
        )
        ax.tick_params(axis="both", which="major", labelsize=8)
        ax.tick_params(axis="both", which="minor", labelsize=6)
        ax.legend()
        fig = ax.get_figure()
        fig.suptitle(f"CDF Comparison: {feature_name}")
        return fig
