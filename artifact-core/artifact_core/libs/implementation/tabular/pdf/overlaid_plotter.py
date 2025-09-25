from typing import Dict, List

import pandas as pd
from matplotlib.figure import Figure

from artifact_core.libs.utils.plotters.overlaid_pdf_plotter import (
    OverlaidPDFConfig,
    OverlaidPDFPlotter,
)
from artifact_core.libs.utils.plotters.overlaid_pmf_plotter import (
    OverlaidPMFConfig,
    OverlaidPMFPlotter,
)
from artifact_core.libs.utils.plotters.plot_combiner import PlotCombinationConfig, PlotCombiner


class TabularOverlaidPDFPlotter:
    _pmf_config = OverlaidPMFConfig(
        plot_color_a="olive",
        plot_color_b="crimson",
        axis_font_size="14",
        label_a="Real",
        label_b="Synthetic",
        cat_density_alpha_a=0.8,
        cat_density_alpha_b=0.8,
        cat_pmf_bar_width=0.4,
        rotation="vertical",
    )
    _pdf_config = OverlaidPDFConfig(
        plot_color_a="olive",
        plot_color_b="crimson",
        gridline_color="black",
        gridline_style=":",
        minor_ax_grid_linewidth=0.1,
        major_ax_grid_linewidth=1.0,
        axis_font_size="14",
        label_a="Real",
        label_b="Synthetic",
        cts_density_n_bins=50,
        cts_density_enable_kde=True,
        cts_densitiy_alpha_a=0.8,
        cts_densitiy_alpha_b=0.4,
        xtick_minor_labelsize=6.0,
        xtick_major_labelsize=8.0,
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
        combined_title="Probability Density Function Comparison",
        combined_title_vertical_position=1.0,
    )

    @classmethod
    def get_overlaid_pdf_plot(
        cls,
        dataset_real: pd.DataFrame,
        dataset_synthetic: pd.DataFrame,
        ls_features_order: List[str],
        ls_cts_features: List[str],
        ls_cat_features: List[str],
        cat_unique_map: Dict[str, List[str]],
    ) -> Figure:
        dict_plots = cls.get_overlaid_pdf_plot_collection(
            dataset_real=dataset_real,
            dataset_synthetic=dataset_synthetic,
            ls_features_order=ls_features_order,
            ls_cts_features=ls_cts_features,
            ls_cat_features=ls_cat_features,
            cat_unique_map=cat_unique_map,
        )
        combined_plot = PlotCombiner.combine(
            dict_plots=dict_plots, config=cls._plot_combiner_config
        )
        return combined_plot

    @classmethod
    def get_overlaid_pdf_plot_collection(
        cls,
        dataset_real: pd.DataFrame,
        dataset_synthetic: pd.DataFrame,
        ls_features_order: List[str],
        ls_cts_features: List[str],
        ls_cat_features: List[str],
        cat_unique_map: Dict[str, List[str]],
    ) -> Dict[str, Figure]:
        dict_plots = {}
        for feature in ls_features_order:
            if feature in ls_cat_features:
                fig = OverlaidPMFPlotter.plot_overlaid_pmf(
                    sr_data_a=dataset_real[feature],
                    sr_data_b=dataset_synthetic[feature],
                    feature_name=feature,
                    ls_unique_categories=cat_unique_map.get(feature, []),
                    config=cls._pmf_config,
                )
            elif feature in ls_cts_features:
                fig = OverlaidPDFPlotter.plot_overlaid_pdf(
                    sr_data_a=dataset_real[feature],
                    sr_data_b=dataset_synthetic[feature],
                    feature_name=feature,
                    config=cls._pdf_config,
                )
            else:
                continue
            dict_plots[feature] = fig
        return dict_plots
