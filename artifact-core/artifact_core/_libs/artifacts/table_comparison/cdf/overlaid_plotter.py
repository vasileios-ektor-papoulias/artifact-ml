from typing import Mapping, Sequence

import pandas as pd
from matplotlib.figure import Figure

from artifact_core._libs.tools.plotters.overlaid_cdf_plotter import (
    OverlaidCDFConfig,
    OverlaidCDFPlotter,
)
from artifact_core._libs.tools.plotters.plot_combiner import PlotCombinationConfig, PlotCombiner


class TabularOverlaidCDFPlotter:
    _cdf_config = OverlaidCDFConfig(
        label_a="Real",
        plot_color_a="olive",
        plot_marker_size_a=5.0,
        plot_marker_edge_width_a=1,
        line_width_a=5.0,
        line_alpha_a=0.5,
        label_b="Synthetic",
        plot_color_b="crimson",
        plot_marker_size_b=5.0,
        plot_marker_edge_width_b=1.0,
        line_width_b=5,
        line_alpha_b=0.5,
        line_style="-",
        plot_marker="o",
        gridline_color="black",
        gridline_style=":",
        minor_ax_grid_linewidth=0.1,
        major_ax_grid_linewidth=1.0,
        axis_font_size="14",
    )
    _plot_combiner_config = PlotCombinationConfig(
        n_cols=3,
        dpi=150,
        figsize_horizontal_multiplier=6.0,
        figsize_vertical_multiplier=4,
        tight_layout_rect=(0, 0, 1, 0.95),
        tight_layout_pad=0.1,
        subplots_adjust_hspace=0.1,
        subplots_adjust_wspace=0.1,
        fig_title_fontsize=5.0,
        include_fig_titles=False,
        combined_title="Cumulative Density Function Comparison",
        combined_title_vertical_position=1.0,
    )

    @classmethod
    def get_overlaid_cdf_plot(
        cls,
        dataset_real: pd.DataFrame,
        dataset_synthetic: pd.DataFrame,
        cts_features: Sequence[str],
    ) -> Figure:
        plots = cls.get_overlaid_cdf_plot_collection(
            dataset_real=dataset_real,
            dataset_synthetic=dataset_synthetic,
            cts_features=cts_features,
        )
        combined_plot = PlotCombiner.combine(plots=plots, config=cls._plot_combiner_config)
        return combined_plot

    @classmethod
    def get_overlaid_cdf_plot_collection(
        cls,
        dataset_real: pd.DataFrame,
        dataset_synthetic: pd.DataFrame,
        cts_features: Sequence[str],
    ) -> Mapping[str, Figure]:
        dict_plots = {}
        for feature in cts_features:
            cdf_plot = OverlaidCDFPlotter.plot_overlaid_cdf(
                sr_data_a=dataset_real[feature],
                sr_data_b=dataset_synthetic[feature],
                feature_name=feature,
                config=cls._cdf_config,
            )
            dict_plots[feature] = cdf_plot
        return dict_plots
