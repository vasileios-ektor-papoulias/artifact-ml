from dataclasses import dataclass
from typing import List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure


@dataclass(frozen=True)
class OverlaidPMFConfig:
    plot_color_a: str = "olive"
    plot_color_b: str = "crimson"
    axis_font_size: str = "14"
    label_a: str = "A"
    label_b: str = "B"
    cat_density_alpha_a: float = 0.8
    cat_density_alpha_b: float = 0.8
    cat_pmf_bar_width: float = 0.4
    rotation: str = "vertical"


class OverlaidPMFPlotter:
    _ylabel = "Probability"

    @classmethod
    def plot_overlaid_pmf(
        cls,
        sr_data_a: pd.Series,
        sr_data_b: pd.Series,
        feature_name: Optional[str] = None,
        unique_categories: Optional[Sequence[str]] = None,
        config: OverlaidPMFConfig = OverlaidPMFConfig(),
    ) -> Figure:
        if unique_categories is None:
            unique_categories = cls._get_ls_unique_categories(
                sr_data_a=sr_data_a, sr_data_b=sr_data_b
            )
        title = cls._get_title(feature_name=feature_name)
        xlabel = cls._get_xlabel(feature_name=feature_name)
        bar_centerpoints = np.arange(len(unique_categories))
        bar_width = config.cat_pmf_bar_width
        sr_freq_a = cls._get_sr_freq(sr_data=sr_data_a, unique_categories=unique_categories)
        sr_freq_b = cls._get_sr_freq(sr_data=sr_data_b, unique_categories=unique_categories)
        fig, ax = plt.subplots()
        plt.close(fig)
        ax.bar(
            x=bar_centerpoints - bar_width / 2,
            height=sr_freq_a.values,  # type: ignore
            width=bar_width,
            color=config.plot_color_a,
            alpha=config.cat_density_alpha_a,
            label=config.label_a,
        )
        ax.bar(
            x=bar_centerpoints + bar_width / 2,
            height=sr_freq_b.values,  # type: ignore
            width=bar_width,
            color=config.plot_color_b,
            alpha=config.cat_density_alpha_b,
            label=config.label_b,
        )
        ax.set_xticks(ticks=bar_centerpoints)
        ax.set_xticklabels(labels=unique_categories, rotation=config.rotation)
        ax.set_xlabel(xlabel=xlabel, fontsize=config.axis_font_size)
        ax.set_ylabel(ylabel=cls._ylabel, fontsize=config.axis_font_size)
        ax.legend()
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
            return f"PMF Comparison: {feature_name}"
        return "PMF Comparison"

    @staticmethod
    def _get_xlabel(feature_name: Optional[str] = None) -> str:
        if feature_name is not None:
            return feature_name
        return "Categories"

    @staticmethod
    def _get_ls_unique_categories(sr_data_a: pd.Series, sr_data_b: pd.Series) -> List[str]:
        ls_unique_a = sr_data_a.astype(str).unique().tolist()
        ls_unique_b = sr_data_b.astype(str).unique().tolist()
        ls_unique_categories = list(set(ls_unique_a).union(ls_unique_b))
        return ls_unique_categories
