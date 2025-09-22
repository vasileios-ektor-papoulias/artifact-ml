from dataclasses import dataclass
from typing import Dict, Hashable, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.image import AxesImage

from artifact_core.libs.implementation.binary_classification.confusion.calculator import (
    ConfusionCalculator,
)
from artifact_core.libs.implementation.binary_classification.confusion.normalized_calculator import (
    ConfusionNormalizationStrategy,
    NormalizedConfusionCalculator,
)


@dataclass(frozen=True)
class ConfusionMatrixPlotConfig:
    title: Optional[str] = "Confusion Matrix"
    dpi: int = 120
    show_values: bool = True
    value_fmt: str = ".2f"


class ConfusionMatrixPlotter:
    @classmethod
    def plot_multiple(
        cls,
        true: Mapping[Hashable, str],
        predicted: Mapping[Hashable, str],
        pos_label: str,
        neg_label: str,
        normalization_types: Sequence[ConfusionNormalizationStrategy],
        config: ConfusionMatrixPlotConfig = ConfusionMatrixPlotConfig(),
    ) -> Dict[ConfusionNormalizationStrategy, Figure]:
        plot_collection = {
            normalization: cls.plot(
                true=true,
                predicted=predicted,
                pos_label=pos_label,
                neg_label=neg_label,
                normalization=normalization,
                config=config,
            )
            for normalization in normalization_types
        }
        return plot_collection

    @classmethod
    def plot(
        cls,
        true: Mapping[Hashable, str],
        predicted: Mapping[Hashable, str],
        pos_label: str,
        neg_label: str,
        normalization: ConfusionNormalizationStrategy,
        config: ConfusionMatrixPlotConfig = ConfusionMatrixPlotConfig(),
    ) -> Figure:
        arr_cm_raw = ConfusionCalculator.compute_confusion_matrix(
            true=true, predicted=predicted, pos_label=pos_label, neg_label=neg_label
        ).astype(float)
        arr_cm = NormalizedConfusionCalculator.compute_normalized_confusion_matrix(
            true=true,
            predicted=predicted,
            pos_label=pos_label,
            neg_label=neg_label,
            normalization=normalization,
        )
        fig, ax = cls._make_figure(dpi=config.dpi)
        im = cls._draw_semantic_matrix(ax=ax, cm=arr_cm)
        cls._decorate_axes(ax=ax, title=config.title, tick_labels=(pos_label, neg_label))
        cls._add_colorbar(fig=fig, ax=ax, im=im)
        if config.show_values:
            cls._annotate_cells(
                ax=ax,
                cm=arr_cm,
                raw_cm=arr_cm_raw,
                normalized=normalization is not ConfusionNormalizationStrategy.NONE,
                value_fmt=config.value_fmt,
            )
        fig.tight_layout()
        plt.close(fig)
        return fig

    @staticmethod
    def _make_figure(dpi: int) -> Tuple[Figure, Axes]:
        fig, ax = plt.subplots(figsize=(5, 4), dpi=dpi)
        return fig, ax

    @staticmethod
    def _draw_matrix(ax: Axes, cm: np.ndarray, cmap: str):
        im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
        return im

    @staticmethod
    def _decorate_axes(
        ax: Axes,
        title: Optional[str],
        tick_labels: Tuple[str, str],
    ) -> None:
        ax.set_title(title or "")
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_xticks([0, 1], labels=list(tick_labels))
        ax.set_yticks([0, 1], labels=list(tick_labels))
        ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
        ax.grid(which="minor", color="w", linestyle="-", linewidth=2, alpha=0.5)
        ax.tick_params(axis="both", which="both", length=0)

    @staticmethod
    def _add_colorbar(fig: Figure, ax: Axes, im):
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    @staticmethod
    def _annotate_cells(
        ax: Axes,
        cm: np.ndarray,
        raw_cm: np.ndarray,
        normalized: bool,
        value_fmt: str,
    ) -> None:
        thresh = cm.max() / 2.0 if cm.size else 0.0
        for i in range(2):
            for j in range(2):
                text = format(cm[i, j], value_fmt) if normalized else f"{int(raw_cm[i, j])}"
                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=10,
                )

    @staticmethod
    def _draw_semantic_matrix(ax: Axes, cm: np.ndarray) -> AxesImage:
        sign = np.array([[+1.0, -1.0], [-1.0, +1.0]], dtype=float)
        signed = cm * sign
        max_abs = float(np.max(np.abs(signed))) if signed.size else 1.0
        if max_abs == 0:
            max_abs = 1.0
        cmap = LinearSegmentedColormap.from_list("bad_to_good", ["red", "white", "green"], N=256)
        norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)
        im = ax.imshow(signed, cmap=cmap, norm=norm, interpolation="nearest")
        return im
