from dataclasses import dataclass
from typing import Dict, Hashable, Iterable, Mapping, Optional

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from artifact_core.libs.implementation.binary_classification.score_distribution.partitioner import (
    BinarySampleSplit,
)
from artifact_core.libs.implementation.binary_classification.score_distribution.sampler import (
    ScoreDistributionSampler,
)
from artifact_core.libs.implementation.tabular.pdf.overlaid_plotter import OverlaidPDFPlotter
from artifact_core.libs.implementation.tabular.pdf.plotter import PDFPlotter
from artifact_core.libs.utils.plot_combiner import PlotCombinationConfig


@dataclass(frozen=True)
class ScoreDistributionPlotterConfig:
    prob_col_name: str = "P(y=positive)"
    single_title_prefix: Optional[str] = None
    label_positive: str = "Positive (true)"
    label_negative: str = "Negative (true)"
    label_all: str = "All"


class ScorePDFPlotter:
    @classmethod
    def plot(
        cls,
        id_to_is_pos: Mapping[Hashable, bool],
        id_to_prob_pos: Mapping[Hashable, float],
        split: BinarySampleSplit,
        config: ScoreDistributionPlotterConfig = ScoreDistributionPlotterConfig(),
    ) -> Figure:
        sample = ScoreDistributionSampler.get_sample(
            id_to_is_pos=id_to_is_pos,
            id_to_prob_pos=id_to_prob_pos,
            split=split,
        )
        col_name = cls._get_col_name(config=config, split=split)
        plot = cls._sample_to_pdf(sample=sample, col_name=col_name)
        return plot

    @classmethod
    def plot_multiple(
        cls,
        id_to_is_pos: Mapping[Hashable, bool],
        id_to_prob_pos: Mapping[Hashable, float],
        splits: Iterable[BinarySampleSplit],
        config: ScoreDistributionPlotterConfig = ScoreDistributionPlotterConfig(),
    ) -> Dict[BinarySampleSplit, Figure]:
        samples_by_split = ScoreDistributionSampler.get_dict_samples(
            id_to_is_pos=id_to_is_pos,
            id_to_prob_pos=id_to_prob_pos,
            splits=list(splits),
        )
        dict_plots: Dict[BinarySampleSplit, Figure] = {}
        for split, samples in samples_by_split.items():
            col = cls._get_col_name(config=config, split=split)
            dict_plots[split] = cls._sample_to_pdf(sample=samples, col_name=col)
        return dict_plots

    @classmethod
    def plot_overlaid(
        cls,
        id_to_is_pos: Mapping[Hashable, bool],
        id_to_prob_pos: Mapping[Hashable, float],
        config: ScoreDistributionPlotterConfig = ScoreDistributionPlotterConfig(),
    ) -> Figure:
        pos_samples = ScoreDistributionSampler.get_sample(
            id_to_is_pos=id_to_is_pos,
            id_to_prob_pos=id_to_prob_pos,
            split=BinarySampleSplit.POSITIVE,
        )
        neg_samples = ScoreDistributionSampler.get_sample(
            id_to_is_pos=id_to_is_pos,
            id_to_prob_pos=id_to_prob_pos,
            split=BinarySampleSplit.NEGATIVE,
        )
        col_name = config.prob_col_name
        df_pos = pd.DataFrame({col_name: np.asarray(pos_samples, dtype=float)})
        df_neg = pd.DataFrame({col_name: np.asarray(neg_samples, dtype=float)})
        fig = _ScoreOverlaidPDFPlotter.get_overlaid_pdf_plot(
            dataset_real=df_pos,
            dataset_synthetic=df_neg,
            ls_features_order=[col_name],
            ls_cts_features=[col_name],
            ls_cat_features=[],
            cat_unique_map={},
        )
        return fig

    @staticmethod
    def _sample_to_pdf(sample: Iterable[float], col_name: str) -> Figure:
        arr = np.asarray(list(sample), dtype=float)
        df = pd.DataFrame({col_name: arr})
        pdf_plot = _ScorePDFPlotter.get_pdf_plot(
            dataset=df,
            ls_features_order=[col_name],
            ls_cts_features=[col_name],
            ls_cat_features=[],
            cat_unique_map={},
        )
        return pdf_plot

    @staticmethod
    def _get_col_name(config: ScoreDistributionPlotterConfig, split: BinarySampleSplit) -> str:
        prefix = (config.single_title_prefix + ": ") if config.single_title_prefix else ""
        if split is BinarySampleSplit.ALL:
            suffix = config.label_all
        elif split is BinarySampleSplit.POSITIVE:
            suffix = config.label_positive
        elif split is BinarySampleSplit.NEGATIVE:
            suffix = config.label_negative
        else:
            suffix = split.value
        return f"{prefix}{config.prob_col_name} — {suffix}"


class _ScoreOverlaidPDFPlotter(OverlaidPDFPlotter):
    _label_real = "Positive (true)"
    _label_synthetic = "Negative (true)"
    _plot_color_real = "tab:blue"
    _plot_color_synthetic = "tab:orange"
    _cts_densitiy_alpha_real = 0.8
    _cts_densitiy_alpha_synthetic = 0.5
    _cts_density_n_bins = 70
    _plot_combiner_config = PlotCombinationConfig(
        n_cols=1,
        dpi=150,
        figsize_horizontal_multiplier=6,
        figsize_vertical_multiplier=4,
        tight_layout_rect=(0, 0, 1, 0.95),
        tight_layout_pad=0.1,
        subplots_adjust_hspace=0.1,
        subplots_adjust_wspace=0.1,
        fig_title_fontsize=5,
        include_fig_titles=False,
        combined_title="Score PDF: Positive Class vs Negative Class",
        combined_title_vertical_position=1,
    )


class _ScorePDFPlotter(PDFPlotter):
    _plot_color = "tab:blue"
    _gridline_color = "grey"
    _gridline_style = "--"
    _axis_font_size = "12"
    _minor_ax_grid_linewidth = 0.2
    _major_ax_grid_linewidth = 0.8
    _cts_density_enable_kde = True
    _cts_densitiy_alpha = 0.6
    _cts_density_n_bins = 70
    _plot_combiner_config = PlotCombinationConfig(
        n_cols=1,
        dpi=150,
        figsize_horizontal_multiplier=6,
        figsize_vertical_multiplier=4,
        tight_layout_rect=(0, 0, 1, 0.95),
        tight_layout_pad=0.1,
        subplots_adjust_hspace=0.1,
        subplots_adjust_wspace=0.1,
        fig_title_fontsize=5,
        include_fig_titles=False,
        combined_title="Score PDF",
        combined_title_vertical_position=1,
    )
