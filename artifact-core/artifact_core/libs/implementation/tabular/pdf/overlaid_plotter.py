from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from artifact_core.libs.utils.plot_combiner import (
    PlotCombinationConfig,
    PlotCombiner,
)


class OverlaidPDFPlotter:
    _plot_color_real = "olive"
    _plot_color_synthetic = "crimson"
    _gridline_color = "black"
    _gridline_style = ":"
    _minor_ax_grid_linewidth = 0.1
    _major_ax_grid_linewidth = 1
    _axis_font_size = "14"

    _cat_densitiy_alpha_real = 0.8
    _cat_densitiy_alpha_synthetic = 0.8
    _cat_pmf_bar_width = 0.4

    _cts_density_n_bins = 50
    _cts_density_enable_kde = True
    _cts_densitiy_alpha_real = 0.8
    _cts_densitiy_alpha_synthetic = 0.4

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
        combined_title="Probability Density Function Comparison",
        combined_title_vertical_position=1,
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
                fig = cls._plot_overlaid_pdf_categorical(
                    sr_data_real=dataset_real[feature],
                    sr_data_synthetic=dataset_synthetic[feature],
                    feature_name=feature,
                    ls_unique_categories=cat_unique_map.get(feature, []),
                )
            elif feature in ls_cts_features:
                fig = cls._plot_overlaid_pdf_continuous(
                    sr_data_real=dataset_real[feature],
                    sr_data_synthetic=dataset_synthetic[feature],
                    feature_name=feature,
                )
            else:
                continue
            dict_plots[feature] = fig
        return dict_plots

    @classmethod
    def _plot_overlaid_pdf_categorical(
        cls,
        sr_data_real: pd.Series,
        sr_data_synthetic: pd.Series,
        feature_name: str,
        ls_unique_categories: List[str],
    ) -> Figure:
        freq_real = sr_data_real.value_counts(normalize=True).reindex(
            index=ls_unique_categories, fill_value=0
        )
        freq_synth = sr_data_synthetic.value_counts(normalize=True).reindex(
            index=ls_unique_categories, fill_value=0
        )

        fig, ax = plt.subplots()
        plt.close(fig)
        positions = np.arange(len(ls_unique_categories))
        bar_width = cls._cat_pmf_bar_width
        ax.bar(
            positions - bar_width / 2,
            freq_real.values,
            bar_width,
            color=cls._plot_color_real,
            alpha=cls._cat_densitiy_alpha_real,
            label="Real",
        )
        ax.bar(
            positions + bar_width / 2,
            freq_synth.values,
            bar_width,
            color=cls._plot_color_synthetic,
            alpha=cls._cat_densitiy_alpha_synthetic,
            label="Synthetic",
        )
        ax.set_xticks(positions)
        ax.set_xticklabels(ls_unique_categories, rotation="vertical")
        ax.set_xlabel(feature_name, fontsize=cls._axis_font_size)
        ax.set_ylabel("Probability", fontsize=cls._axis_font_size)
        ax.legend()
        fig.suptitle(f"PMF Comparison: {feature_name}")
        return fig

    @classmethod
    def _plot_overlaid_pdf_continuous(
        cls,
        sr_data_real: pd.Series,
        sr_data_synthetic: pd.Series,
        feature_name: str,
    ) -> Figure:
        fig, ax = plt.subplots()
        plt.close(fig)
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
            sr_data_real.dropna(),
            bins=cls._cts_density_n_bins,
            stat="density",
            color=cls._plot_color_real,
            alpha=cls._cts_densitiy_alpha_real,
            kde=cls._cts_density_enable_kde,
            label="Real",
            ax=ax,
        )
        sns.histplot(
            sr_data_synthetic.dropna(),
            bins=cls._cts_density_n_bins,
            stat="density",
            color=cls._plot_color_synthetic,
            alpha=cls._cts_densitiy_alpha_synthetic,
            kde=cls._cts_density_enable_kde,
            label="Synthetic",
            ax=ax,
        )
        ax.set_xlabel("Values", fontsize=cls._axis_font_size)
        ax.set_ylabel("Probability Density", fontsize=cls._axis_font_size)
        ax.tick_params(axis="both", which="major", labelsize=8)
        ax.tick_params(axis="both", which="minor", labelsize=6)
        ax.legend()
        fig.suptitle(f"PDF Comparison: {feature_name}")
        return fig
