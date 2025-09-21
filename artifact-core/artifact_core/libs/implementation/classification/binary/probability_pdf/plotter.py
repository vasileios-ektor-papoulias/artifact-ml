from dataclasses import dataclass
from enum import Enum
from typing import Dict, Hashable, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from artifact_core.libs.implementation.tabular.pdf.overlaid_plotter import OverlaidPDFPlotter
from artifact_core.libs.implementation.tabular.pdf.plotter import PDFPlotter


class PositiveProbabilitySlice(Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    ALL = "ALL"


@dataclass(frozen=True)
class BinaryProbDensityDelegateConfig:
    prob_col_name: str = "P(y=positive)"
    single_title_prefix: Optional[str] = None
    label_positive: str = "Positive (true)"
    label_negative: str = "Negative (true)"
    label_all: str = "All"


class PredictedProbabilityPlotter:
    @classmethod
    def plot_density_for(
        cls,
        true: Mapping[Hashable, str],
        probs: Mapping[Hashable, float],
        pos_label: str,
        slice_type: PositiveProbabilitySlice,
        config: BinaryProbDensityDelegateConfig = BinaryProbDensityDelegateConfig(),
    ) -> Figure:
        y_true, y_prob = cls._align_labels(true, probs)
        arr = cls._get_prob_slice(
            y_true=y_true, y_prob=y_prob, pos_label=pos_label, slice_type=slice_type
        )
        col = cls._get_column_name(config, slice_type)
        df = pd.DataFrame({col: arr})
        fig = PDFPlotter.get_pdf_plot(
            dataset=df,
            ls_features_order=[col],
            ls_cts_features=[col],
            ls_cat_features=[],
            cat_unique_map={},
        )
        return fig

    @classmethod
    def build_density_plots_dict(
        cls,
        true: Mapping[Hashable, str],
        probs: Mapping[Hashable, float],
        pos_label: str,
        ls_slice_types: Iterable[PositiveProbabilitySlice],
        config: BinaryProbDensityDelegateConfig = BinaryProbDensityDelegateConfig(),
    ) -> Dict[PositiveProbabilitySlice, Figure]:
        y_true, y_prob = cls._align_labels(true, probs)
        dict_figures: Dict[PositiveProbabilitySlice, Figure] = {}
        for slice in ls_slice_types:
            arr = cls._get_prob_slice(
                y_true=y_true, y_prob=y_prob, pos_label=pos_label, slice_type=slice
            )
            col = cls._get_column_name(config=config, slice=slice)
            df = pd.DataFrame({col: arr})
            fig = PDFPlotter.get_pdf_plot(
                dataset=df,
                ls_features_order=[col],
                ls_cts_features=[col],
                ls_cat_features=[],
                cat_unique_map={},
            )
            dict_figures[slice] = fig
        return dict_figures

    @classmethod
    def plot_overlaid_pos_neg(
        cls,
        true: Mapping[Hashable, str],
        probs: Mapping[Hashable, float],
        pos_label: str,
        config: BinaryProbDensityDelegateConfig = BinaryProbDensityDelegateConfig(),
    ) -> Figure:
        y_true, y_prob = cls._align_labels(true, probs)
        pos_arr = cls._get_prob_slice(y_true, y_prob, pos_label, PositiveProbabilitySlice.POSITIVE)
        neg_arr = cls._get_prob_slice(y_true, y_prob, pos_label, PositiveProbabilitySlice.NEGATIVE)
        col = config.prob_col_name
        df_pos = pd.DataFrame({col: pos_arr})
        df_neg = pd.DataFrame({col: neg_arr})
        fig = OverlaidPDFPlotter.get_overlaid_pdf_plot(
            dataset_real=df_pos,
            dataset_synthetic=df_neg,
            ls_features_order=[col],
            ls_cts_features=[col],
            ls_cat_features=[],
            cat_unique_map={},
        )
        return fig

    @classmethod
    def _get_column_name(
        cls, config: BinaryProbDensityDelegateConfig, slice: PositiveProbabilitySlice
    ) -> str:
        prefix = (config.single_title_prefix + ": ") if config.single_title_prefix else ""
        return f"{prefix}{config.prob_col_name} — {slice.value}"

    @classmethod
    def _get_prob_slice(
        cls,
        y_true: List[str],
        y_prob: List[float],
        pos_label: str,
        slice_type: PositiveProbabilitySlice,
    ) -> np.ndarray:
        y_true_arr = np.array(y_true, dtype=object)
        y_prob_arr = np.array(y_prob, dtype=float)

        if slice_type is PositiveProbabilitySlice.ALL:
            return y_prob_arr
        elif slice_type is PositiveProbabilitySlice.POSITIVE:
            return y_prob_arr[y_true_arr == pos_label]
        elif slice_type is PositiveProbabilitySlice.NEGATIVE:
            return y_prob_arr[y_true_arr != pos_label]
        else:
            raise ValueError(f"Unknown slice: {slice_type}")

    @classmethod
    def _align_labels(
        cls,
        true: Mapping[Hashable, str],
        probs: Mapping[Hashable, float],
    ) -> Tuple[List[str], List[float]]:
        missing = [k for k in true if k not in probs]
        if missing:
            raise KeyError(
                f"Probabilities missing for {len(missing)} id(s): "
                f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
            )
        keys = list(true.keys())
        y_true = [true[k] for k in keys]
        y_prob = [float(probs[k]) for k in keys]
        return y_true, y_prob
