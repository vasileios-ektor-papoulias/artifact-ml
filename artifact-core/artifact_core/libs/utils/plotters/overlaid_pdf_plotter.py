from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure


@dataclass(frozen=True)
class OverlaidPDFConfig:
    plot_color_a: str = "olive"
    plot_color_b: str = "crimson"
    gridline_color: str = "black"
    gridline_style: str = ":"
    minor_ax_grid_linewidth: float = 0.1
    major_ax_grid_linewidth: float = 1.0
    axis_font_size: str = "14"
    label_a: str = "A"
    label_b: str = "B"
    cts_density_n_bins: int = 50
    cts_density_enable_kde: bool = True
    cts_densitiy_alpha_a: float = 0.8
    cts_densitiy_alpha_b: float = 0.4
    xtick_minor_labelsize: float = 6.0
    xtick_major_labelsize: float = 8.0


class OverlaidPDFPlotter:
    @classmethod
    def plot_overlaid_pdf(
        cls,
        sr_data_a: pd.Series,
        sr_data_b: pd.Series,
        feature_name: Optional[str] = None,
        config: OverlaidPDFConfig = OverlaidPDFConfig(),
    ) -> Figure:
        title = cls._get_title(feature_name=feature_name)
        fig, ax = plt.subplots()
        plt.close(fig)
        sns.histplot(
            data=sr_data_a.dropna(),  # type: ignore
            bins=config.cts_density_n_bins,
            stat="density",
            color=config.plot_color_a,
            alpha=config.cts_densitiy_alpha_a,
            kde=config.cts_density_enable_kde,
            label=config.label_a,
            ax=ax,
        )
        sns.histplot(
            data=sr_data_b.dropna(),  # type: ignore
            bins=config.cts_density_n_bins,
            stat="density",
            color=config.plot_color_b,
            alpha=config.cts_densitiy_alpha_b,
            kde=config.cts_density_enable_kde,
            label=config.label_b,
            ax=ax,
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
        ax.set_xlabel(xlabel="Values", fontsize=config.axis_font_size)
        ax.set_ylabel(ylabel="Probability Density", fontsize=config.axis_font_size)
        ax.tick_params(axis="both", which="minor", labelsize=config.xtick_minor_labelsize)
        ax.tick_params(axis="both", which="major", labelsize=config.xtick_major_labelsize)
        ax.set_axisbelow(b=True)
        ax.legend()
        fig.suptitle(t=title)
        return fig

    @staticmethod
    def _get_title(feature_name: Optional[str] = None) -> str:
        if feature_name is not None:
            return f"PDF Comparison: {feature_name}"
        return "PDF Comparison"
