from dataclasses import dataclass, field
from typing import List, Mapping, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


@dataclass(frozen=True)
class PlotCombinationConfig:
    n_cols: int = 2
    dpi: int = 150
    figsize_horizontal_multiplier: float = 6.0
    figsize_vertical_multiplier: float = 4.0
    tight_layout_rect: Tuple[float, float, float, float] = field(
        default_factory=lambda: (0, 0, 1, 0.95)
    )
    tight_layout_pad: float = 0.1
    subplots_adjust_hspace: float = 0.1
    subplots_adjust_wspace: float = 0.1
    include_fig_titles: bool = False
    fig_title_fontsize: float = 5.0
    combined_title: Optional[str] = None
    combined_title_vertical_position: float = 1.0


class PlotCombiner:
    @classmethod
    def combine(cls, plots: Mapping[str, Figure], config: PlotCombinationConfig) -> Figure:
        nfigs = len(plots)
        if nfigs == 0:
            combined_fig = cls._get_empty_figure(
                figsize_horizontal_multiplier=config.figsize_horizontal_multiplier,
                figsize_vertical_multiplier=config.figsize_vertical_multiplier,
                dpi=config.dpi,
            )
        else:
            combined_fig = cls._get_combined_figure(plots=plots, config=config)
        if config.combined_title is not None:
            combined_fig.suptitle(
                t=config.combined_title,
                y=config.combined_title_vertical_position,
            )
        return combined_fig

    @classmethod
    def _get_combined_figure(
        cls, plots: Mapping[str, Figure], config: PlotCombinationConfig
    ) -> Figure:
        combined_fig, axs = cls._create_combined_subplots(plots, config)
        cls._draw_all_content(axs=axs, plots=plots, config=config)
        cls._turn_off_extra_axes(axs=axs, plots=plots)
        cls._apply_layout_adjustments(fig=combined_fig, config=config)
        return combined_fig

    @classmethod
    def _create_combined_subplots(
        cls, plots: Mapping[str, Figure], config: PlotCombinationConfig
    ) -> Tuple[Figure, List[Axes]]:
        nfigs = len(plots)
        nrows = (nfigs + config.n_cols - 1) // config.n_cols
        combined_fig, raw_axs = plt.subplots(
            nrows=nrows,
            ncols=config.n_cols,
            figsize=(
                config.n_cols * config.figsize_horizontal_multiplier,
                nrows * config.figsize_vertical_multiplier,
            ),
            dpi=config.dpi,
        )
        plt.close(combined_fig)
        axs_array = np.atleast_1d(raw_axs).flatten()
        axs: List[Axes] = axs_array.tolist()
        return combined_fig, axs

    @classmethod
    def _draw_all_content(
        cls,
        axs: List[Axes],
        plots: Mapping[str, Figure],
        config: PlotCombinationConfig,
    ):
        for ax, (label, fig) in zip(axs, plots.items()):
            cls._draw_component_figure_content(ax=ax, label=label, fig=fig, config=config)

    @classmethod
    def _turn_off_extra_axes(cls, axs: List[Axes], plots: Mapping[str, Figure]):
        for extra_ax in axs[len(plots) :]:
            extra_ax.axis("off")

    @classmethod
    def _apply_layout_adjustments(cls, fig: Figure, config: PlotCombinationConfig):
        fig.tight_layout(rect=config.tight_layout_rect, pad=config.tight_layout_pad)
        fig.subplots_adjust(
            hspace=config.subplots_adjust_hspace,
            wspace=config.subplots_adjust_wspace,
        )

    @staticmethod
    def _draw_component_figure_content(
        ax: Axes, label: str, fig: Figure, config: PlotCombinationConfig
    ):
        if not isinstance(fig.canvas, FigureCanvasAgg):
            raise RuntimeError("PlotCombiner only supports Agg backend.")
        fig.tight_layout()
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        renderer = fig.canvas.get_renderer()
        image = np.frombuffer(renderer.buffer_rgba(), dtype=np.uint8).reshape((height, width, 4))
        ax.imshow(image)
        if config.include_fig_titles:
            ax.set_title(label, fontsize=config.fig_title_fontsize)
        ax.axis("off")

    @staticmethod
    def _get_empty_figure(
        figsize_horizontal_multiplier: float,
        figsize_vertical_multiplier: float,
        dpi: int,
    ) -> Figure:
        empty_fig = plt.figure(
            figsize=(
                figsize_horizontal_multiplier,
                figsize_vertical_multiplier,
            ),
            dpi=dpi,
        )

        plt.close(empty_fig)
        return empty_fig
