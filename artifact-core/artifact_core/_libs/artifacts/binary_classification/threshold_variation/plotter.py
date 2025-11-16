from dataclasses import dataclass
from enum import Enum
from typing import Hashable, Literal, Mapping, Sequence, Tuple

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

from artifact_core._base.typing.artifact_result import Array
from artifact_core._utils.collections.map_aligner import MapAligner

ThresholdVariationCurveTypeLiteral = Literal[
    "ROC", "PR", "DET", "TPR_THRESHOLD", "PRECISION_THRESHOLD"
]


class ThresholdVariationCurveType(Enum):
    ROC = "roc"  # Receiver Operator Characteristic/ Recall vs FPR
    PR = "pr"  # Precision vs Recall
    DET = "det"  # Decision Error Tradeoff/ FNR vs FPR
    RECALL_THRESHOLD = "recall_threshold"  # Recall vs Decision Threshold
    PRECISION_THRESHOLD = "precision_threshold"  # Precision vs Decision Threshold


@dataclass(frozen=True)
class ThresholdVariationCurvePlotterConfig:
    dpi: int = 120
    figsize: Tuple[float, float] = (6.0, 4.5)
    linewidth: float = 2.0
    alpha: float = 1.0
    color: str = "blue"
    show_baseline: bool = True
    linestyle_baseline: str = "--"
    linewidth_baseline: float = 1.0
    alpha_baseline: float = 1.0
    color_baseline: str = "black"


class ThresholdVariationCurvePlotter:
    @classmethod
    def plot_multiple(
        cls,
        curve_types: Sequence[ThresholdVariationCurveType],
        true: Mapping[Hashable, bool],
        probs: Mapping[Hashable, float],
        config: ThresholdVariationCurvePlotterConfig = ThresholdVariationCurvePlotterConfig(),
    ) -> Mapping[ThresholdVariationCurveType, Figure]:
        dict_plots = {
            curve_type: cls.plot(curve_type=curve_type, true=true, probs=probs, config=config)
            for curve_type in curve_types
        }
        return dict_plots

    @classmethod
    def plot(
        cls,
        curve_type: ThresholdVariationCurveType,
        true: Mapping[Hashable, bool],
        probs: Mapping[Hashable, float],
        config: ThresholdVariationCurvePlotterConfig = ThresholdVariationCurvePlotterConfig(),
    ) -> Figure:
        _, y_true, y_probs = MapAligner.align(left=true, right=probs)
        y_true_bin = np.asarray(y_true, dtype=int)
        y_probs = np.asarray(y_probs, dtype=float)
        if curve_type is ThresholdVariationCurveType.ROC:
            fig = cls._plot_roc(y_true_bin=y_true_bin, y_probs=y_probs, config=config)
        elif curve_type is ThresholdVariationCurveType.PR:
            fig = cls._plot_pr(y_true_bin=y_true_bin, y_probs=y_probs, config=config)
        elif curve_type is ThresholdVariationCurveType.DET:
            fig = cls._plot_det(y_true_bin=y_true_bin, y_probs=y_probs, config=config)
        elif curve_type is ThresholdVariationCurveType.RECALL_THRESHOLD:
            fig = cls._plot_recall_threshold(y_true_bin=y_true_bin, y_probs=y_probs, config=config)
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
        y_true_bin: Array,
        y_probs: Array,
        config: ThresholdVariationCurvePlotterConfig,
    ) -> Figure:
        if not cls._has_both_classes(y_true_bin=y_true_bin):
            return cls._empty_fig(
                config=config,
                note="Receiver Operator Characteristic (ROC) Curve (needs both classes)",
            )
        fpr, tpr, _ = roc_curve(y_true=y_true_bin, y_score=y_probs)
        auc = float(roc_auc_score(y_true=y_true_bin, y_score=y_probs))
        fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
        ax.plot(
            fpr,
            tpr,
            lw=config.linewidth,
            color=config.color,
            alpha=config.alpha,
            label=f"ROC (AUC={auc:.3f})",
        )
        if config.show_baseline:
            ax.plot(
                [0, 1],
                [0, 1],
                linestyle=config.linestyle_baseline,
                color=config.color_baseline,
                lw=config.linewidth_baseline,
                alpha=config.alpha_baseline,
                label="Baseline",
            )
        cls._decorate_axes(
            ax,
            title="Receiver Operator Characteristic (ROC) Curve",
            x="False Positive Rate (FPR)",
            y="Recall/ True Positive Rate (TPR)",
        )
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.05)
        ax.legend(loc="lower right")
        fig.tight_layout()
        return fig

    @classmethod
    def _plot_pr(
        cls,
        y_true_bin: Array,
        y_probs: Array,
        config: ThresholdVariationCurvePlotterConfig,
    ) -> Figure:
        precision, recall, _ = precision_recall_curve(y_true_bin, y_probs)
        pr_auc = float(average_precision_score(y_true=y_true_bin, y_score=y_probs))
        fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
        ax.plot(
            recall,
            precision,
            lw=config.linewidth,
            color=config.color,
            alpha=config.alpha,
            label=f"PR (PR-AUC (AP)={pr_auc:.3f})",
        )
        if config.show_baseline:
            prevalence = y_true_bin.mean()
            ax.hlines(
                prevalence,
                xmin=0,
                xmax=1,
                linestyle=config.linestyle_baseline,
                colors=config.color_baseline,
                lw=config.linewidth_baseline,
                alpha=config.alpha_baseline,
                label=f"Baseline (π={prevalence:.2f})",
            )
        cls._decorate_axes(ax, title="Precision–Recall Curve", x="Recall", y="Precision")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.05)
        ax.legend(loc="lower left")
        fig.tight_layout()
        return fig

    @classmethod
    def _plot_det(
        cls,
        y_true_bin: Array,
        y_probs: Array,
        config: ThresholdVariationCurvePlotterConfig,
    ) -> Figure:
        if not cls._has_both_classes(y_true_bin=y_true_bin):
            return cls._empty_fig(
                config=config, note="Decision Error Tradeoff (DET) Curve (needs both classes)"
            )
        fpr, fnr, _ = det_curve(y_true=y_true_bin, y_score=y_probs)
        fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
        ax.plot(fpr, fnr, lw=config.linewidth, color=config.color, alpha=config.alpha, label="DET")
        if config.show_baseline:
            ax.plot(
                [0, 1],
                [1, 0],
                linestyle=config.linestyle_baseline,
                color=config.color_baseline,
                lw=config.linewidth_baseline,
                alpha=config.alpha_baseline,
                label="Baseline",
            )
        cls._decorate_axes(
            ax,
            title="Decision Error Tradeoff (DET) Curve",
            x="False Positive Rate (FPR)",
            y="False Negative Rate (FNR)",
        )
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.05)
        ax.legend(loc="upper right")
        fig.tight_layout()
        return fig

    @classmethod
    def _plot_recall_threshold(
        cls,
        y_true_bin: Array,
        y_probs: Array,
        config: ThresholdVariationCurvePlotterConfig,
    ) -> Figure:
        if not cls._has_both_classes(y_true_bin=y_true_bin):
            return cls._empty_fig(config=config, note="Recall-Threshold Curve (needs both classes)")
        _, tpr, thr = roc_curve(y_true=y_true_bin, y_score=y_probs)
        thr = thr[1:]
        tpr = tpr[1:]
        fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
        ax.plot(
            thr,
            tpr,
            lw=config.linewidth,
            color=config.color,
            alpha=config.alpha,
            label="Recall-Threshold",
        )
        if config.show_baseline:
            ax.plot(
                [0, 1],
                [1, 0],
                linestyle=config.linestyle_baseline,
                color=config.color_baseline,
                lw=config.linewidth_baseline,
                alpha=config.alpha_baseline,
                label="Baseline",
            )
        cls._decorate_axes(
            ax,
            title="Recall-Threshold Curve",
            x="Decision Threshold",
            y="Recall",
        )
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.05)
        ax.legend(loc="best")
        fig.tight_layout()
        return fig

    @classmethod
    def _plot_precision_threshold(
        cls,
        y_true_bin: Array,
        y_probs: Array,
        config: ThresholdVariationCurvePlotterConfig,
    ) -> Figure:
        precision, _, thr = precision_recall_curve(y_true_bin, y_probs)
        fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
        ax.plot(
            thr,
            precision[1:],
            lw=config.linewidth,
            color=config.color,
            alpha=config.alpha,
            label="Precision-Threshold",
        )
        if config.show_baseline:
            prevalence = y_true_bin.mean()
            ax.hlines(
                prevalence,
                xmin=0,
                xmax=1,
                linestyle=config.linestyle_baseline,
                colors=config.color_baseline,
                lw=config.linewidth_baseline,
                alpha=config.alpha_baseline,
                label=f"Baseline (π={prevalence:.2f})",
            )
        cls._decorate_axes(
            ax,
            title="Precision-Threshold Curve",
            x="Decision Threshold",
            y="Precision",
        )
        ax.set_xlim(0.0, 1.0)
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
    def _has_both_classes(y_true_bin: Array) -> bool:
        has_true = np.any(y_true_bin == 1)
        has_false = np.any(y_true_bin == 0)
        has_both = bool(has_true and has_false)
        return has_both

    @staticmethod
    def _empty_fig(config: ThresholdVariationCurvePlotterConfig, note: str) -> Figure:
        fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
        ax.text(0.5, 0.5, note, ha="center", va="center", fontsize=11, alpha=0.8)
        ax.axis("off")
        fig.tight_layout()
        return fig
