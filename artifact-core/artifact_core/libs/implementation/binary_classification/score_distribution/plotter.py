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


@dataclass(frozen=True)
class ScoreDistributionPlotterConfig:
    prob_col_name: str = "P(y=positive)"
    single_title_prefix: Optional[str] = None
    label_positive: str = "Positive (true)"
    label_negative: str = "Negative (true)"
    label_all: str = "All"


class ScoreDistributionPlotter:
    @classmethod
    def plot_density_for(
        cls,
        id_to_is_pos: Mapping[Hashable, bool],
        id_to_prob_pos: Mapping[Hashable, float],
        split: BinarySampleSplit,
        config: ScoreDistributionPlotterConfig = ScoreDistributionPlotterConfig(),
    ) -> Figure:
        samples = ScoreDistributionSampler.get_sample(
            id_to_is_pos=id_to_is_pos,
            id_to_prob_pos=id_to_prob_pos,
            split=split,
        )
        col_name = cls._get_col_name(config=config, split=split)
        return cls._samples_to_pdf(samples=samples, col_name=col_name)

    @classmethod
    def build_density_plots_dict(
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
        figures: Dict[BinarySampleSplit, Figure] = {}
        for split, samples in samples_by_split.items():
            col = cls._get_col_name(config=config, split=split)
            figures[split] = cls._samples_to_pdf(samples=samples, col_name=col)
        return figures

    @classmethod
    def plot_overlaid_pos_neg(
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
        fig = OverlaidPDFPlotter.get_overlaid_pdf_plot(
            dataset_real=df_pos,
            dataset_synthetic=df_neg,
            ls_features_order=[col_name],
            ls_cts_features=[col_name],
            ls_cat_features=[],
            cat_unique_map={},
        )
        return fig

    @staticmethod
    def _samples_to_pdf(samples: Iterable[float], col_name: str) -> Figure:
        arr = np.asarray(list(samples), dtype=float)
        df = pd.DataFrame({col_name: arr})
        pdf_plot = PDFPlotter.get_pdf_plot(
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
        if split is BinarySampleSplit.NONE:
            suffix = config.label_all
        elif split is BinarySampleSplit.POSITIVE:
            suffix = config.label_positive
        elif split is BinarySampleSplit.NEGATIVE:
            suffix = config.label_negative
        else:
            suffix = split.value
        return f"{prefix}{config.prob_col_name} — {suffix}"
