import math
from dataclasses import dataclass
from typing import List, Tuple

from artifact_core.libs.utils.autoscale.base import (
    PlotAutoscaleArgs,
    PlotAutoscaleHyperparams,
    PlotAutoscaler,
    PlotScale,
)
from artifact_core.libs.utils.plot_combiner import PlotCombinationConfig


@dataclass(frozen=True)
class CombinedAutoscalerArgs(PlotAutoscaleArgs):
    num_subplots: int
    ls_subplot_dims: List[Tuple[float, float]]


@dataclass(frozen=True)
class CombinedAutoscalerHyperparams(PlotAutoscaleHyperparams):
    min_scale_factor: float
    max_scale_factor: float


@dataclass(frozen=True)
class CombinedPlotScale(PlotScale):
    n_cols: int
    title_scale: float


class CombinedPlotAutoscaler(
    PlotAutoscaler[
        CombinedAutoscalerArgs,
        CombinedAutoscalerHyperparams,
        CombinedPlotScale,
    ]
):
    @classmethod
    def compute(
        cls, args: CombinedAutoscalerArgs, params: CombinedAutoscalerHyperparams
    ) -> CombinedPlotScale:
        n_cols = cls._get_n_cols(n_subplots=args.num_subplots)
        raw_width_scale = cls._get_raw_scale(ls_dims=[dims[0] for dims in args.ls_subplot_dims])
        raw_height_scale = cls._get_raw_scale(ls_dims=[dims[1] for dims in args.ls_subplot_dims])
        width_scale = cls._clamp_scale_factor(
            value=raw_width_scale,
            min_factor=params.min_scale_factor,
            max_factor=params.max_scale_factor,
        )
        height_scale = cls._clamp_scale_factor(
            value=raw_height_scale,
            min_factor=params.min_scale_factor,
            max_factor=params.max_scale_factor,
        )
        size_factor = math.sqrt(width_scale * height_scale)
        title_scale = max(params.min_scale_factor, size_factor)
        scale = CombinedPlotScale(
            figure_width_scale=width_scale,
            figure_height_scale=height_scale,
            n_cols=n_cols,
            title_scale=title_scale,
        )
        return scale

    @staticmethod
    def get_scaled_combiner_config(
        num_plots: int,
        ls_subplot_dims: List[Tuple[float, float]],
        base_config: PlotCombinationConfig,
        scale_config: CombinedAutoscalerHyperparams,
    ) -> PlotCombinationConfig:
        args = CombinedAutoscalerArgs(num_subplots=num_plots, ls_subplot_dims=ls_subplot_dims)
        scales = CombinedPlotAutoscaler.compute(args=args, params=scale_config)
        scaled_horizontal = scales.figure_width_scale
        scaled_vertical = scales.figure_height_scale
        scaled_fig_title_fontsize = base_config.fig_title_fontsize * scales.title_scale
        scaled_combined_title_fontsize = base_config.combined_title_fontsize * scales.title_scale
        config = PlotCombinationConfig(
            n_cols=scales.n_cols,
            dpi=base_config.dpi,
            figsize_horizontal_multiplier=scaled_horizontal,
            figsize_vertical_multiplier=scaled_vertical,
            tight_layout_rect=base_config.tight_layout_rect,
            tight_layout_pad=base_config.tight_layout_pad,
            subplots_adjust_hspace=base_config.subplots_adjust_hspace,
            subplots_adjust_wspace=base_config.subplots_adjust_wspace,
            include_fig_titles=base_config.include_fig_titles,
            fig_title_fontsize=scaled_fig_title_fontsize,
            combined_title=base_config.combined_title,
            combined_title_fontsize=scaled_combined_title_fontsize,
            combined_title_vertical_position=(base_config.combined_title_vertical_position),
        )
        return config

    @classmethod
    def _get_raw_scale(cls, ls_dims: List[float]) -> float:
        average = sum(ls_dims) / len(ls_dims)
        maximum = max(ls_dims)
        return (average + maximum) / 2

    @staticmethod
    def _get_n_cols(n_subplots: int) -> int:
        if n_subplots <= 3:
            n_cols = 3
        elif n_subplots <= 8:
            n_cols = 4
        elif n_subplots <= 15:
            n_cols = 5
        elif n_subplots <= 24:
            n_cols = 6
        else:
            n_cols = min(8, max(3, int(math.sqrt(n_subplots))))
        return n_cols
