from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.figure import Figure, SubFigure

from artifact_core._base.typing.artifact_result import Array


@dataclass(frozen=True)
class OverlaidCDFConfig:
    label_a: str = "A"
    plot_color_a: str = "olive"
    plot_marker_size_a: float = 5.0
    plot_marker_edge_width_a: float = 1.0
    line_width_a: float = 5.0
    line_alpha_a: float = 0.5
    label_b: str = "B"
    plot_color_b: str = "crimson"
    plot_marker_size_b: float = 5.0
    plot_marker_edge_width_b: float = 1.0
    line_width_b: float = 5.0
    line_alpha_b: float = 0.5
    line_style: str = "-"
    plot_marker: str = "o"
    gridline_color: str = "black"
    gridline_style: str = ":"
    minor_ax_grid_linewidth: float = 0.1
    major_ax_grid_linewidth: float = 1.0
    axis_font_size: str = "14"


class OverlaidCDFPlotter:
    _xlabel = "Values"
    _ylabel = "Normalized Cumulative Sum"

    @classmethod
    def plot_overlaid_cdf(
        cls,
        sr_data_a: pd.Series,
        sr_data_b: pd.Series,
        feature_name: Optional[str] = None,
        config: OverlaidCDFConfig = OverlaidCDFConfig(),
    ) -> Figure:
        title = cls._get_title(feature_name=feature_name)
        x1, y1 = cls._get_data(sr_data=sr_data_a)
        x2, y2 = cls._get_data(sr_data=sr_data_b)
        fig, ax = plt.subplots()
        plt.close(fig)
        ax.plot(
            x1,
            y1,
            marker=config.plot_marker,
            markersize=config.plot_marker_size_a,
            markeredgewidth=config.plot_marker_edge_width_a,
            linestyle=config.line_style,
            linewidth=config.line_width_a,
            alpha=config.line_alpha_a,
            color=config.plot_color_a,
            label=config.label_a,
        )
        ax.plot(
            x2,
            y2,
            marker=config.plot_marker,
            markersize=config.plot_marker_size_b,
            markeredgewidth=config.plot_marker_edge_width_b,
            linestyle=config.line_style,
            linewidth=config.line_width_b,
            alpha=config.line_alpha_b,
            color=config.plot_color_b,
            label=config.label_b,
        )
        ax.grid(
            visible=True,
            which="minor",
            linestyle=config.gridline_style,
            linewidth=config.minor_ax_grid_linewidth,
            color=config.gridline_color,
        )
        ax.grid(
            visible=True,
            which="major",
            linestyle=config.gridline_style,
            linewidth=config.major_ax_grid_linewidth,
            color=config.gridline_color,
        )
        ax.set_axisbelow(b=True)
        ax.minorticks_on()
        ax.tick_params(axis="both", which="major", labelsize=8)
        ax.tick_params(axis="both", which="minor", labelsize=6)
        ax.set_xlabel(xlabel=cls._xlabel, size=config.axis_font_size)
        ax.set_ylabel(ylabel=cls._ylabel, size=config.axis_font_size)
        ax.legend()
        fig = ax.get_figure()
        if fig is None:
            return Figure()
        if isinstance(fig, SubFigure):
            return Figure()
        fig.suptitle(t=title)
        return fig

    @staticmethod
    def _get_data(sr_data: pd.Series) -> Tuple[Array, Array]:
        x = sr_data.sort_values()
        y = np.arange(1, len(sr_data) + 1) / len(sr_data)
        return (x, y)  # type: ignore

    @staticmethod
    def _get_title(feature_name: Optional[str] = None) -> str:
        if feature_name is not None:
            return f"CDF Comparison: {feature_name}"
        return "CDF Comparison"
