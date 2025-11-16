from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from artifact_core._base.typing.artifact_result import Array
from artifact_core._libs.artifacts.table_comparison.descriptive_stats.calculator import (
    DescriptiveStatistic,
    TableStatsCalculator,
)
from artifact_core._libs.tools.plotters.plot_combiner import PlotCombinationConfig, PlotCombiner


@dataclass(frozen=True)
class StatsAlignmentPlotterConfig:
    line_color: str = "olive"
    line_width: float = 3.0
    line_alpha: float = 0.5
    scatter_marker_color: str = "chocolate"
    scatter_marker_size: float = 40.0
    scatter_marker_shape: str = "o"
    scatter_marker_alpha: float = 1.0


class DescriptiveStatsAlignmentPlotter:
    _plot_config = StatsAlignmentPlotterConfig()
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
    _stats: Sequence[DescriptiveStatistic] = [stat for stat in DescriptiveStatistic]

    @classmethod
    def get_stat_alignment_plot(
        cls,
        dataset_real: pd.DataFrame,
        dataset_synthetic: pd.DataFrame,
        cts_features: Sequence[str],
        stat: DescriptiveStatistic,
    ) -> Figure:
        dict_stats = TableStatsCalculator.compute_juxtaposition(
            df_real=dataset_real,
            df_synthetic=dataset_synthetic,
            cts_features=cts_features,
            stat=stat,
        )
        if not dict_stats:
            plot = cls._get_empty_figure()
        else:
            arr_descriptive_stats = np.stack(list(dict_stats.values()))
            stats_real = arr_descriptive_stats[:, 0]
            stats_synthetic = arr_descriptive_stats[:, 1]
            plot = cls._get_stat_alignment_plot(
                stats_real=stats_real,
                stats_synthetic=stats_synthetic,
                stat_name=stat.value,
                plot_config=cls._plot_config,
            )
        return plot

    @classmethod
    def get_combined_stat_alignment_plot(
        cls,
        dataset_real: pd.DataFrame,
        dataset_synthetic: pd.DataFrame,
        cts_features: Sequence[str],
    ) -> Figure:
        plots = cls._get_stat_alignment_plot_collection(
            dataset_real=dataset_real,
            dataset_synthetic=dataset_synthetic,
            ls_cts_features=cts_features,
            stats=cls._stats,
        )
        combined_plot = PlotCombiner.combine(plots=plots, config=cls._plot_combiner_config)
        return combined_plot

    @classmethod
    def _get_stat_alignment_plot_collection(
        cls,
        dataset_real: pd.DataFrame,
        dataset_synthetic: pd.DataFrame,
        ls_cts_features: Sequence[str],
        stats: Sequence[DescriptiveStatistic],
    ) -> Mapping[str, Figure]:
        dict_plots = {}
        for stat in stats:
            plot = cls.get_stat_alignment_plot(
                dataset_real=dataset_real,
                dataset_synthetic=dataset_synthetic,
                cts_features=ls_cts_features,
                stat=stat,
            )
            dict_plots[stat.name] = plot
        return dict_plots

    @classmethod
    def _get_stat_alignment_plot(
        cls,
        stats_real: Array,
        stats_synthetic: Array,
        stat_name: str,
        plot_config: StatsAlignmentPlotterConfig = StatsAlignmentPlotterConfig(),
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
            color=plot_config.line_color,
            linewidth=plot_config.line_width,
            alpha=plot_config.line_alpha,
        )
        sns.scatterplot(
            x=stats_real_abs_log,
            y=stats_synthetic_abs_log,
            ax=ax,
            color=plot_config.scatter_marker_color,
            s=plot_config.scatter_marker_size,
            marker=plot_config.scatter_marker_shape,
            alpha=plot_config.scatter_marker_alpha,
        )
        ax.set_title(label=f"{stat_name} comparison")
        ax.set_xlabel(xlabel=f"real data {stat_name} (log-abs)")
        ax.set_ylabel(ylabel=f"synthetic data {stat_name} (log-abs)")
        return fig

    @staticmethod
    def _convert_to_log_scale(stats: Array) -> Array:
        stats_abs_log = np.log(np.add(abs(stats), 1e-5))
        return stats_abs_log

    @staticmethod
    def _get_empty_figure():
        empty_fig = plt.figure(figsize=(6, 4), dpi=150)
        plt.close(empty_fig)
        return empty_fig
