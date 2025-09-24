from typing import Dict, List

import pandas as pd
from matplotlib.figure import Figure

from artifact_core.libs.utils.plotters.cdf_plotter import CDFConfig, CDFPlotter
from artifact_core.libs.utils.plotters.plot_combiner import (
    PlotCombinationConfig,
    PlotCombiner,
)


class TabularCDFPlotter:
    _cdf_config = CDFConfig(
        plot_color="olive",
        plot_marker_size=5,
        plot_marker_edge_width=1,
        line_width=5,
        line_alpha=0.5,
        plot_marker="o",
        line_style="-",
        gridline_color="black",
        gridline_style=":",
        minor_ax_grid_linewidth=0.1,
        major_ax_grid_linewidth=1,
        axis_font_size="14",
    )
    _plot_combiner_config = PlotCombinationConfig(
        n_cols=3,
        dpi=150,
        figsize_horizontal_multiplier=6.0,
        figsize_vertical_multiplier=4.0,
        tight_layout_rect=(0, 0, 1, 0.95),
        tight_layout_pad=0.1,
        subplots_adjust_hspace=0.1,
        subplots_adjust_wspace=0.1,
        fig_title_fontsize=5.0,
        include_fig_titles=False,
        combined_title="Cumulative Density Functions",
        combined_title_vertical_position=1.0,
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
            cdf_plot = CDFPlotter.plot_cdf(
                sr_data=dataset[feature], feature_name=feature, config=cls._cdf_config
            )
            dict_plots[feature] = cdf_plot
        return dict_plots
