from typing import Dict, List

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from artifact_core.libs.implementation.descriptive_statistics.calculator import (
    DescriptiveStatistic,
    DescriptiveStatisticsCalculator,
)
from artifact_core.libs.utils.plot_combiner import (
    PlotCombinationConfig,
    PlotCombiner,
)


class DescriptiveStatComparisonPlotter:
    _plot_combiner_config = PlotCombinationConfig(
        n_cols=2,
        dpi=150,
        figsize_horizontal_multiplier=6,
        figsize_vertical_multiplier=4,
        tight_layout_rect=(0, 0, 1, 0.95),
        tight_layout_pad=0.1,
        subplots_adjust_hspace=0.1,
        subplots_adjust_wspace=0.1,
        fig_title_fontsize=5,
        include_fig_titles=False,
        combined_title="Descriptive Statistics Comparison",
    )
    _line_color = "olive"
    _line_width = 3
    _line_alpha = 0.5
    _scatter_marker_color = "chocolate"
    _scatter_marker_size = 40
    _scatter_marker_shape = "o"
    _scatter_marker_alpha = 1
    _ls_stats = [stat for stat in DescriptiveStatistic]

    @classmethod
    def get_combined_stat_comparison_plot(
        cls,
        dataset_real: pd.DataFrame,
        dataset_synthetic: pd.DataFrame,
        ls_cts_features: List[str],
    ) -> Figure:
        dict_plots = cls._get_stat_comparison_plot_collection(
            dataset_real=dataset_real,
            dataset_synthetic=dataset_synthetic,
            ls_cts_features=ls_cts_features,
            ls_stats=cls._ls_stats,
        )
        combined_plot = PlotCombiner.combine(
            dict_plots=dict_plots, config=cls._plot_combiner_config
        )
        return combined_plot

    @classmethod
    def get_stat_comparison_plot(
        cls,
        dataset_real: pd.DataFrame,
        dataset_synthetic: pd.DataFrame,
        ls_cts_features: List[str],
        stat: DescriptiveStatistic,
    ) -> Figure:
        dict_stats = DescriptiveStatisticsCalculator.compute_juxtaposition(
            df_real=dataset_real,
            df_synthetic=dataset_synthetic,
            ls_cts_features=ls_cts_features,
            stat=stat,
        )
        if not dict_stats:
            plot = cls._get_empty_figure()
        else:
            arr_descriptive_stats = np.stack(list(dict_stats.values()))
            stats_real = arr_descriptive_stats[:, 0]
            stats_synthetic = arr_descriptive_stats[:, 1]
            plot = cls._plot_stat_comparison(
                stats_real=stats_real,
                stats_synthetic=stats_synthetic,
                stat_name=stat.value,
            )
        return plot

    @classmethod
    def _get_stat_comparison_plot_collection(
        cls,
        dataset_real: pd.DataFrame,
        dataset_synthetic: pd.DataFrame,
        ls_cts_features: List[str],
        ls_stats: List[DescriptiveStatistic],
    ) -> Dict[str, Figure]:
        dict_plots = {}
        for stat in ls_stats:
            plot = cls.get_stat_comparison_plot(
                dataset_real=dataset_real,
                dataset_synthetic=dataset_synthetic,
                ls_cts_features=ls_cts_features,
                stat=stat,
            )
            dict_plots[stat.name] = plot
        return dict_plots

    @classmethod
    def _plot_stat_comparison(
        cls,
        stats_real: np.ndarray,
        stats_synthetic: np.ndarray,
        stat_name: str,
    ) -> Figure:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        plt.close(fig)
        fig.suptitle(
            f"Log-Abs {stat_name.capitalize()} of Continuous Features\n",
            fontsize=16,
        )
        ax.grid(True)
        stats_real_abs_log = cls._convert_to_log_scale(stats=stats_real)
        stats_synthetic_abs_log = cls._convert_to_log_scale(stats=stats_synthetic)
        min_val = min(stats_real_abs_log) - 1
        max_val = max(stats_real_abs_log) + 1
        line = np.arange(min_val, max_val)
        sns.lineplot(
            x=line,
            y=line,
            ax=ax,
            color=cls._line_color,
            linewidth=cls._line_width,
            alpha=cls._line_alpha,
        )
        sns.scatterplot(
            x=stats_real_abs_log,
            y=stats_synthetic_abs_log,
            ax=ax,
            color=cls._scatter_marker_color,
            s=cls._scatter_marker_size,
            marker=cls._scatter_marker_shape,
            alpha=cls._scatter_marker_alpha,
        )

        ax.set_title(f"{stat_name} comparison")
        ax.set_xlabel(f"real data {stat_name} (log-abs)")
        ax.set_ylabel(f"synthetic data {stat_name} (log-abs)")
        return fig

    @staticmethod
    def _convert_to_log_scale(stats: np.ndarray) -> np.ndarray:
        stats_abs_log = np.log(np.add(abs(stats), 1e-5))
        return stats_abs_log

    @staticmethod
    def _get_empty_figure():
        empty_fig = plt.figure(figsize=(6, 4), dpi=150)
        plt.close(empty_fig)
        return empty_fig
