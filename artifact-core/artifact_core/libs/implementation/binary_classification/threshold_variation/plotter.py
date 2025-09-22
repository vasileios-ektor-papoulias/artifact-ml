from dataclasses import dataclass
from enum import Enum
from typing import Dict, Hashable, List, Literal, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.metrics import (
    average_precision_score,
    det_curve,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

ThresholdVariationCurveTypeLiteral = Literal[
    "ROC", "PR", "DET", "TPR_THRESHOLD", "PRECISION_THRESHOLD"
]


class ThresholdVariationCurveType(Enum):
    ROC = "roc"  # TPR vs FPR
    PR = "pr"  # Precision vs Recall
    DET = "det"  # FNR vs FPR
    TPR_THRESHOLD = "tpr_threshold"  # TPR vs threshold
    PRECISION_THRESHOLD = "precision_threshold"  # Precision vs threshold


@dataclass(frozen=True)
class ThresholdVariationCurvePlotterConfig:
    title: Optional[str] = None
    dpi: int = 120
    figsize: Tuple[float, float] = (6.0, 4.5)
    linewidth: float = 2.0
    alpha: float = 0.9
    show_chance: bool = True  # used for ROC (and optionally PR baseline)


class ThresholdVariationCurvePlotter:
    @classmethod
    def plot_multiple(
        cls,
        curve_types: Sequence[ThresholdVariationCurveType],
        true: Mapping[Hashable, str],
        probs: Mapping[Hashable, float],
        pos_label: str,
        config: ThresholdVariationCurvePlotterConfig = ThresholdVariationCurvePlotterConfig(),
    ) -> Dict[ThresholdVariationCurveType, Figure]:
        dict_plots = {
            curve_type: cls.plot(
                curve_type=curve_type, true=true, probs=probs, pos_label=pos_label, config=config
            )
            for curve_type in curve_types
        }
        return dict_plots

    @classmethod
    def plot(
        cls,
        curve_type: ThresholdVariationCurveType,
        true: Mapping[Hashable, str],
        probs: Mapping[Hashable, float],
        pos_label: str,
        config: ThresholdVariationCurvePlotterConfig = ThresholdVariationCurvePlotterConfig(),
    ) -> Figure:
        y_true, y_probs = cls._align_labels(true=true, probs=probs)
        y_true_bin = (np.array(y_true) == pos_label).astype(int)
        if curve_type is ThresholdVariationCurveType.ROC:
            fig = cls._plot_roc(y_true_bin=y_true_bin, y_probs=y_probs, config=config)
        elif curve_type is ThresholdVariationCurveType.PR:
            fig = cls._plot_pr(y_true_bin=y_true_bin, y_probs=y_probs, config=config)
        elif curve_type is ThresholdVariationCurveType.DET:
            fig = cls._plot_det(y_true_bin=y_true_bin, y_probs=y_probs, config=config)
        elif curve_type is ThresholdVariationCurveType.TPR_THRESHOLD:
            fig = cls._plot_tpr_threshold(y_true_bin=y_true_bin, y_probs=y_probs, config=config)
        elif curve_type is ThresholdVariationCurveType.PRECISION_THRESHOLD:
            fig = cls._plot_precision_threshold(
                y_true_bin=y_true_bin, y_probs=y_probs, config=config
            )
        else:
            raise ValueError(f"Unsupported curve type: {curve_type}")
        plt.close(fig)
        return fig

    @classmethod
    def _plot_roc(
        cls,
        y_true_bin: List[int],
        y_probs: List[float],
        config: ThresholdVariationCurvePlotterConfig,
    ) -> Figure:
        fpr, tpr, _ = roc_curve(y_true=y_true_bin, y_score=y_probs)
        auc = float(roc_auc_score(y_true=y_true_bin, y_score=y_probs))
        fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
        if config.show_chance:
            ax.plot([0, 1], [0, 1], linestyle="--", lw=1.0, alpha=0.7, label="chance")
        ax.plot(fpr, tpr, lw=config.linewidth, alpha=config.alpha, label=f"ROC (AUC={auc:.3f})")
        cls._decorate_axes(
            ax, title=config.title or "ROC Curve", x="False Positive Rate", y="True Positive Rate"
        )
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.05)
        ax.legend(loc="lower right")
        fig.tight_layout()
        return fig

    @classmethod
    def _plot_pr(
        cls,
        y_true_bin: List[int],
        y_probs: List[float],
        config: ThresholdVariationCurvePlotterConfig,
    ) -> Figure:
        precision, recall, _ = precision_recall_curve(y_true_bin, y_probs)
        ap = float(average_precision_score(y_true=y_true_bin, y_score=y_probs))
        fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
        ax.plot(
            recall, precision, lw=config.linewidth, alpha=config.alpha, label=f"PR (AP={ap:.3f})"
        )
        cls._decorate_axes(
            ax, title=config.title or "Precision–Recall Curve", x="Recall", y="Precision"
        )
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.05)
        ax.legend(loc="lower left")
        fig.tight_layout()
        return fig

    @classmethod
    def _plot_det(cls, y_true_bin, y_probs, config: ThresholdVariationCurvePlotterConfig) -> Figure:
        fpr, fnr, _ = det_curve(y_true=y_true_bin, y_score=y_probs)
        fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
        ax.plot(fpr, fnr, lw=config.linewidth, alpha=config.alpha, label="DET")
        cls._decorate_axes(
            ax, title=config.title or "DET Curve", x="False Positive Rate", y="False Negative Rate"
        )
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.legend(loc="upper right")
        fig.tight_layout()
        return fig

    @classmethod
    def _plot_tpr_threshold(
        cls,
        y_true_bin: List[int],
        y_probs: List[float],
        config: ThresholdVariationCurvePlotterConfig,
    ) -> Figure:
        _, tpr, thr = roc_curve(y_true=y_true_bin, y_score=y_probs)
        thr = thr[1:]
        tpr = tpr[1:]
        fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
        ax.plot(thr, tpr, lw=config.linewidth, alpha=config.alpha, label="TPR vs threshold")
        cls._decorate_axes(
            ax,
            title=config.title or "TPR vs Threshold",
            x="Decision threshold",
            y="True Positive Rate",
        )
        if thr.size:
            ax.set_xlim(thr.min(), thr.max())
        ax.set_ylim(0.0, 1.05)
        ax.legend(loc="best")
        fig.tight_layout()
        return fig

    @classmethod
    def _plot_precision_threshold(
        cls,
        y_true_bin: List[int],
        y_probs: List[float],
        config: ThresholdVariationCurvePlotterConfig,
    ) -> Figure:
        prec, _, thr = precision_recall_curve(y_true_bin, y_probs)
        fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
        ax.plot(
            thr, prec[1:], lw=config.linewidth, alpha=config.alpha, label="Precision vs threshold"
        )
        cls._decorate_axes(
            ax,
            title=config.title or "Precision vs Threshold",
            x="Decision threshold",
            y="Precision",
        )
        if thr.size:
            ax.set_xlim(thr.min(), thr.max())
        ax.set_ylim(0.0, 1.05)
        ax.legend(loc="best")
        fig.tight_layout()
        return fig

    @staticmethod
    def _decorate_axes(ax: Axes, *, title: str, x: str, y: str) -> None:
        ax.set_title(title)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.6)

    @staticmethod
    def _align_labels(
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
        return [true[k] for k in keys], [float(probs[k]) for k in keys]
