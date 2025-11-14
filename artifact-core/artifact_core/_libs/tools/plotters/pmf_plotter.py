from dataclasses import dataclass
from typing import List, Optional, Sequence

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure


@dataclass(frozen=True)
class PMFConfig:
    plot_color: str = "olive"
    alpha: float = 0.7
    axis_font_size: str = "14"
    rotation: str = "vertical"


class PMFPlotter:
    _ylabel = "Probability"

    @classmethod
    def plot_pmf(
        cls,
        sr_data: pd.Series,
        feature_name: Optional[str] = None,
        unique_categories: Optional[Sequence[str]] = None,
        config: PMFConfig = PMFConfig(),
    ) -> Figure:
        if unique_categories is None:
            unique_categories = cls._get_ls_unique_categories(sr_data=sr_data)
        title = cls._get_title(feature_name=feature_name)
        xlabel = cls._get_xlabel(feature_name=feature_name)

        sr_freq = cls._get_sr_freq(sr_data=sr_data, unique_categories=unique_categories)
        bar_centerpoints = range(len(sr_freq.index))
        fig, ax = plt.subplots()
        plt.close(fig)
        ax.bar(
            x=bar_centerpoints,
            height=sr_freq.values,  # type: ignore
            color=config.plot_color,
            alpha=config.alpha,
        )
        ax.set_xticks(ticks=bar_centerpoints)
        ax.set_xticklabels(labels=sr_freq.index.astype(str), rotation=config.rotation)
        ax.set_xlabel(xlabel=xlabel, fontsize=config.axis_font_size)
        ax.set_ylabel(ylabel=cls._ylabel, fontsize=config.axis_font_size)
        fig.suptitle(t=title)
        return fig

    @staticmethod
    def _get_sr_freq(sr_data: pd.Series, unique_categories: Sequence[str]) -> pd.Series:
        sr_freq = sr_data.value_counts(normalize=True).reindex(
            index=list(unique_categories), fill_value=0
        )
        return sr_freq

    @staticmethod
    def _get_title(feature_name: Optional[str] = None) -> str:
        if feature_name is not None:
            return f"PMF: {feature_name}"
        return "PMF"

    @staticmethod
    def _get_xlabel(feature_name: Optional[str] = None) -> str:
        if feature_name is not None:
            return feature_name
        return "Categories"

    @staticmethod
    def _get_ls_unique_categories(sr_data: pd.Series) -> List[str]:
        return sr_data.astype(str).unique().tolist()  # type: ignore
