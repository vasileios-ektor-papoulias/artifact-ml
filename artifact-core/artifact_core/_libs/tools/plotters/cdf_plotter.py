from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.figure import Figure, SubFigure

from artifact_core._base.typing.artifact_result import Array


@dataclass(frozen=True)
class CDFConfig:
    plot_color: str = "olive"
    plot_marker_size: float = 5.0
    plot_marker_edge_width: float = 1.0
    line_width: float = 5.0
    line_alpha: float = 0.5
    plot_marker: str = "o"
    line_style: str = "-"
    gridline_color: str = "black"
    gridline_style: str = ":"
    minor_ax_grid_linewidth: float = 0.1
    major_ax_grid_linewidth: float = 1.0
    axis_font_size: str = "14"


class CDFPlotter:
    _ylabel = "Normalized Cumulative Sum"

    @classmethod
    def plot_cdf(
        cls, sr_data: pd.Series, feature_name: Optional[str] = None, config: CDFConfig = CDFConfig()
    ) -> Figure:
        title = cls._get_title(feature_name=feature_name)
        x, y = cls._get_data(sr_data=sr_data)
        fig, ax = plt.subplots()
        plt.close(fig)
        ax.plot(
            x,
            y,
            marker=config.plot_marker,
            markersize=config.plot_marker_size,
            markeredgewidth=config.plot_marker_edge_width,
            linestyle=config.line_style,
            linewidth=config.line_width,
            alpha=config.line_alpha,
            color=config.plot_color,
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
        ax.tick_params(axis="both", which="major", labelsize=8)
        ax.tick_params(axis="both", which="minor", labelsize=6)
        ax.set_xlabel(xlabel="Values", size=config.axis_font_size)
        ax.set_ylabel(ylabel=cls._ylabel, size=config.axis_font_size)
        ax.set_axisbelow(b=True)
        plot = ax.get_figure()
        if plot is None:
            return Figure()
        if isinstance(plot, SubFigure):
            return Figure()
        plot.suptitle(t=title)
        return plot

    @staticmethod
    def _get_data(sr_data: pd.Series) -> Tuple[Array, Array]:
        x = sr_data.sort_values()
        y = np.arange(1, len(sr_data) + 1) / len(sr_data)
        return (x, y)  # type: ignore

    @staticmethod
    def _get_title(feature_name: Optional[str] = None) -> str:
        if feature_name is not None:
            return f"CDF: {feature_name}"
        return "CDF"
