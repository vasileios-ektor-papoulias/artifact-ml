from typing import Dict, List

import pandas as pd
from matplotlib.figure import Figure

from artifact_core._libs.artifacts.tools.plotters.pdf_plotter import PDFConfig, PDFPlotter
from artifact_core._libs.artifacts.tools.plotters.plot_combiner import (
    PlotCombinationConfig,
    PlotCombiner,
)
from artifact_core._libs.artifacts.tools.plotters.pmf_plotter import PMFConfig, PMFPlotter


class TabularPDFPlotter:
    _pmf_config = PMFConfig(plot_color="olive", alpha=0.7, axis_font_size="14", rotation="vertical")
    _pdf_config = PDFConfig(
        plot_color="olive",
        gridline_color="black",
        gridline_style=":",
        axis_font_size="14",
        minor_ax_grid_linewidth=0.1,
        major_ax_grid_linewidth=1.0,
        cts_density_n_bins=50,
        cts_density_enable_kde=True,
        cts_densitiy_alpha=0.7,
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
        combined_title="Probability Density Functions",
        combined_title_vertical_position=1.0,
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
        combined_plot = PlotCombiner.combine(
            dict_plots=dict_plots, config=cls._plot_combiner_config
        )
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
                fig = PMFPlotter.plot_pmf(
                    sr_data=dataset[feature],
                    feature_name=feature,
                    ls_unique_categories=cat_unique_map.get(feature, []),
                    config=cls._pmf_config,
                )
            elif feature in ls_cts_features:
                fig = PDFPlotter.plot_pdf(
                    sr_data=dataset[feature], feature_name=feature, config=cls._pdf_config
                )
            else:
                continue
            dict_plots[feature] = fig
        return dict_plots
