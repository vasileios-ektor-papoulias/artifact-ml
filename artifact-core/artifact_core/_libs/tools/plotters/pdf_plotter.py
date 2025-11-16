from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure


@dataclass(frozen=True)
class PDFConfig:
    plot_color: str = "olive"
    gridline_color: str = "black"
    gridline_style: str = ":"
    axis_font_size: str = "14"
    minor_ax_grid_linewidth: float = 0.1
    major_ax_grid_linewidth: float = 1.0
    cts_density_n_bins: int = 50
    cts_density_enable_kde: bool = True
    cts_densitiy_alpha: float = 0.7


class PDFPlotter:
    @classmethod
    def plot_pdf(
        cls, sr_data: pd.Series, feature_name: Optional[str] = None, config: PDFConfig = PDFConfig()
    ) -> Figure:
        title = cls._get_title(feature_name=feature_name)
        fig, ax = plt.subplots()
        plt.close(fig)
        sns.histplot(
            data=sr_data.dropna(),  # type: ignore
            bins=config.cts_density_n_bins,
            stat="density",
            color=config.plot_color,
            alpha=config.cts_densitiy_alpha,
            kde=config.cts_density_enable_kde,
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
        ax.set_axisbelow(b=True)
        fig.suptitle(t=title)
        return fig

    @staticmethod
    def _get_title(feature_name: Optional[str] = None) -> str:
        if feature_name is not None:
            return f"PDF: {feature_name}"
        return "PDF"
