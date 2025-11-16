from dataclasses import dataclass
from typing import Dict, Hashable, Iterable, Mapping, Optional

import pandas as pd
from matplotlib.figure import Figure

from artifact_core._libs.artifacts.binary_classification.score_distribution.partitioner import (
    BinarySampleSplit,
)
from artifact_core._libs.artifacts.binary_classification.score_distribution.sampler import (
    ScoreDistributionSampler,
)
from artifact_core._libs.tools.plotters.overlaid_pdf_plotter import (
    OverlaidPDFConfig,
    OverlaidPDFPlotter,
)
from artifact_core._libs.tools.plotters.pdf_plotter import PDFConfig, PDFPlotter


@dataclass(frozen=True)
class ScoreDistributionPlotterConfig:
    prob_col_name: str = "P(y=positive)"
    single_title_prefix: Optional[str] = None
    label_positive: str = "Positive (true)"
    label_negative: str = "Negative (true)"
    label_all: str = "All"


class ScorePDFPlotter:
    _pdf_config = PDFConfig(
        plot_color="blue",
        gridline_color="grey",
        gridline_style="--",
        axis_font_size="12",
        minor_ax_grid_linewidth=0.2,
        major_ax_grid_linewidth=0.8,
        cts_density_enable_kde=True,
        cts_densitiy_alpha=0.6,
        cts_density_n_bins=70,
    )
    _overlaid_pdf_config = OverlaidPDFConfig(
        label_a="Positive (true)",
        label_b="Negative (true)",
        plot_color_a="green",
        plot_color_b="red",
        cts_densitiy_alpha_a=0.8,
        cts_densitiy_alpha_b=0.5,
        cts_density_n_bins=70,
    )

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
        ls_pos_probs = ScoreDistributionSampler.get_sample(
            id_to_is_pos=id_to_is_pos,
            id_to_prob_pos=id_to_prob_pos,
            split=BinarySampleSplit.POSITIVE,
        )
        ls_neg_probs = ScoreDistributionSampler.get_sample(
            id_to_is_pos=id_to_is_pos,
            id_to_prob_pos=id_to_prob_pos,
            split=BinarySampleSplit.NEGATIVE,
        )
        sr_pos_probs = pd.Series(data=ls_pos_probs)
        sr_neg_probs = pd.Series(data=ls_neg_probs)
        fig = OverlaidPDFPlotter.plot_overlaid_pdf(
            sr_data_a=sr_pos_probs,
            sr_data_b=sr_neg_probs,
            feature_name=config.prob_col_name,
            config=cls._overlaid_pdf_config,
        )
        return fig

    @classmethod
    def _sample_to_pdf(cls, sample: Iterable[float], col_name: str) -> Figure:
        sr_probs = pd.Series(data=list(sample))
        pdf_plot = PDFPlotter.plot_pdf(
            sr_data=sr_probs, feature_name=col_name, config=cls._pdf_config
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
