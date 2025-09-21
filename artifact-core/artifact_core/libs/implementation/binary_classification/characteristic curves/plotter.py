from dataclasses import dataclass
from enum import Enum
from typing import Hashable, List, Mapping, Optional, Tuple

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


class BinaryClassificationCurveType(Enum):
    ROC = "ROC"  # TPR vs FPR
    PR = "PR"  # Precision vs Recall
    DET = "DET"  # FNR vs FPR
    TPR_THRESHOLD = "TPR_THRESHOLD"  # TPR vs threshold
    PRECISION_THRESHOLD = "PRECISION_THRESHOLD"  # Precision vs threshold


@dataclass(frozen=True)
class BinaryClassificationCurvePlotConfig:
    title: Optional[str] = None
    dpi: int = 120
    figsize: Tuple[float, float] = (6.0, 4.5)
    linewidth: float = 2.0
    alpha: float = 0.9
    show_chance: bool = True  # used for ROC (and optionally PR baseline)


class BinaryClassificationCurvePlotter:
    @classmethod
    def plot(
        cls,
        plot_type: BinaryClassificationCurveType,
        true: Mapping[Hashable, str],
        probs: Mapping[Hashable, float],
        pos_label: str,
        config: BinaryClassificationCurvePlotConfig = BinaryClassificationCurvePlotConfig(),
    ) -> Figure:
        y_true, y_score = cls._align_labels(true, probs)
        y_pos = (np.array(y_true) == pos_label).astype(int)
        s = np.array(y_score, dtype=float)
        if BinaryClassificationCurveType is BinaryClassificationCurveType.ROC:
            return cls._plot_roc(y_pos, s, config)
        elif BinaryClassificationCurveType is BinaryClassificationCurveType.PR:
            return cls._plot_pr(y_pos, s, config)
        elif BinaryClassificationCurveType is BinaryClassificationCurveType.DET:
            return cls._plot_det(y_pos, s, config)
        elif BinaryClassificationCurveType is BinaryClassificationCurveType.TPR_THRESHOLD:
            return cls._plot_tpr_threshold(y_pos, s, config)
        elif BinaryClassificationCurveType is BinaryClassificationCurveType.PRECISION_THRESHOLD:
            return cls._plot_precision_threshold(y_pos, s, config)
        else:
            raise ValueError(f"Unsupported curve kind: {plot_type}")

    @classmethod
    def _plot_roc(cls, y_pos, s, config: BinaryClassificationCurvePlotConfig) -> Figure:
        fpr, tpr, _ = roc_curve(y_pos, s)
        auc = float(roc_auc_score(y_pos, s))
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
    def _plot_pr(cls, y_pos, s, config: BinaryClassificationCurvePlotConfig) -> Figure:
        precision, recall, _ = precision_recall_curve(y_pos, s)
        ap = float(average_precision_score(y_pos, s))
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
    def _plot_det(cls, y_pos, s, config: BinaryClassificationCurvePlotConfig) -> Figure:
        fpr, fnr, _ = det_curve(y_pos, s)
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
    def _plot_tpr_threshold(cls, y_pos, s, config: BinaryClassificationCurvePlotConfig) -> Figure:
        fpr, tpr, thr = roc_curve(y_pos, s)
        fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
        ax.plot(thr, tpr[1:], lw=config.linewidth, alpha=config.alpha, label="TPR vs threshold")
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
        cls, y_pos, s, config: BinaryClassificationCurvePlotConfig
    ) -> Figure:
        prec, rec, thr = precision_recall_curve(y_pos, s)
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
