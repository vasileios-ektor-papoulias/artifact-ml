from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure

from artifact_core.libs.utils.plot_combiner import (
    PlotCombinationConfig,
    PlotCombiner,
)


@dataclass(frozen=True)
class ProjectionPlotterConfig:
    scatter_color: str = "olive"
    failed_suffix: str = "Projection failed (rank or numeric issues)."
    figsize: tuple[int, int] = (6, 6)
    title_prefix: str = "2D Projection"
    combined_config: PlotCombinationConfig = PlotCombinationConfig(
        n_cols=2,
        dpi=150,
        figsize_horizontal_multiplier=4,
        figsize_vertical_multiplier=4,
        tight_layout_rect=(0, 0, 1, 0.95),
        tight_layout_pad=0.1,
        subplots_adjust_hspace=0.1,
        subplots_adjust_wspace=0.1,
        fig_title_fontsize=10,
        include_fig_titles=True,
        combined_title="2D Projection Comparison",
        combined_title_vertical_position=1.1,
    )
    combined_plot_real_key = "Real"
    combined_plot_synthetic_key = "Synthetic"


class ProjectionPlotter:
    def __init__(self, config: Optional[ProjectionPlotterConfig]):
        if config is None:
            config = ProjectionPlotterConfig()
        self._config = config

    def get_projection_comparison_plot(
        self,
        dataset_projection_2d_real: Optional[np.ndarray],
        dataset_projection_2d_synthetic: Optional[np.ndarray],
        method_name: str,
    ) -> Figure:
        fig_real = self.get_projection_plot(
            dataset_projection_2d=dataset_projection_2d_real,
            method_name=method_name,
        )
        fig_synth = self.get_projection_plot(
            dataset_projection_2d=dataset_projection_2d_synthetic,
            method_name=method_name,
        )
        dict_plots = {
            self._config.combined_plot_real_key: fig_real,
            self._config.combined_plot_synthetic_key: fig_synth,
        }
        combined_fig = PlotCombiner.combine(
            dict_plots=dict_plots, config=self._config.combined_config
        )
        general_title = self._config.combined_config.combined_title
        combined_fig.suptitle(
            t=f"{general_title}: {method_name}",
            y=self._config.combined_config.combined_title_vertical_position,
        )
        return combined_fig

    def get_projection_plot(
        self, dataset_projection_2d: Optional[np.ndarray], method_name: str
    ) -> Figure:
        self._validate_projection(dataset_projection=dataset_projection_2d)
        fig, ax = plt.subplots(figsize=self._config.figsize)
        plt.close(fig=fig)
        if dataset_projection_2d is None:
            ax.set_title(label=f"{method_name}: {self._config.failed_suffix}")
            ax.axis("off")
        else:
            sns.scatterplot(
                x=dataset_projection_2d[:, 0],
                y=dataset_projection_2d[:, 1],
                color=self._config.scatter_color,
                ax=ax,
            )
            ax.set_xlabel("Dim 1")
            ax.set_ylabel("Dim 2")
            fig.suptitle(f"{self._config.title_prefix}: {method_name}")
        return fig

    @staticmethod
    def _validate_projection(dataset_projection: Optional[np.ndarray]):
        if dataset_projection is not None:
            if dataset_projection.ndim != 2 or dataset_projection.shape[1] != 2:
                raise ValueError(f"Expected shape: (n_samples, 2), got {dataset_projection.shape}.")
