import math
from dataclasses import dataclass

from artifact_core.libs.utils.autoscale.base import (
    PlotAutoscaleArgs,
    PlotAutoscaleHyperparams,
    PlotAutoscaler,
    PlotScale,
)


@dataclass(frozen=True)
class GridAutoscalerArgs(PlotAutoscaleArgs):
    grid_width: int
    grid_height: int


@dataclass(frozen=True)
class GridAutoscalerHyperparams(PlotAutoscaleHyperparams):
    min_scale_factor: float
    max_scale_factor: float
    grid_cells_per_base_size: float
    base_figure_width: float
    base_figure_height: float
    base_font_size: float
    base_title_font_size: float
    base_tick_font_size: float
    base_legend_font_size: float
    base_annotation_font_size: float
    base_marker_size: float
    base_line_width: float
    text_font_scale_multiplier: float
    title_font_scale_multiplier: float
    tick_font_scale_multiplier: float
    legend_font_scale_multiplier: float
    annotation_font_scale_multiplier: float


@dataclass(frozen=True)
class GridPlotScale(PlotScale):
    figure_width_scale: float
    figure_height_scale: float
    figure_width: float
    figure_height: float
    font_size: float
    title_font_size: float
    tick_font_size: float
    legend_font_size: float
    annotation_font_size: float
    marker_size: float
    line_width: float


class GridAutoscaler(PlotAutoscaler[GridAutoscalerArgs, GridAutoscalerHyperparams, GridPlotScale]):
    @classmethod
    def compute(cls, args: GridAutoscalerArgs, params: GridAutoscalerHyperparams) -> GridPlotScale:
        height_scale = cls._compute_scale_factor(
            dimension=args.grid_height,
            base_scale=cls._base_scale,
            grid_cells_per_base_size=params.grid_cells_per_base_size,
        )
        height_scale = cls._clamp_scale_factor(
            value=height_scale,
            min_factor=params.min_scale_factor,
            max_factor=params.max_scale_factor,
        )
        width_scale = cls._compute_scale_factor(
            dimension=args.grid_width,
            base_scale=cls._base_scale,
            grid_cells_per_base_size=params.grid_cells_per_base_size,
        )
        width_scale = cls._clamp_scale_factor(
            value=width_scale,
            min_factor=params.min_scale_factor,
            max_factor=params.max_scale_factor,
        )
        avg_scale = math.sqrt(width_scale * height_scale)
        text_scale = cls._compute_text_scale(
            scale_factor=avg_scale, min_factor=params.min_scale_factor
        )
        text_scale = cls._clamp_scale_factor(
            value=text_scale, min_factor=params.min_scale_factor, max_factor=params.max_scale_factor
        )
        scale = GridPlotScale(
            figure_width_scale=width_scale,
            figure_height_scale=height_scale,
            figure_width=width_scale * params.base_figure_width,
            figure_height=height_scale * params.base_figure_height,
            font_size=params.base_font_size * text_scale * params.text_font_scale_multiplier,
            title_font_size=params.base_title_font_size
            * text_scale
            * params.title_font_scale_multiplier,
            tick_font_size=params.base_tick_font_size
            * text_scale
            * params.tick_font_scale_multiplier,
            legend_font_size=params.base_legend_font_size
            * text_scale
            * params.legend_font_scale_multiplier,
            annotation_font_size=params.base_annotation_font_size
            * text_scale
            * params.annotation_font_scale_multiplier,
            marker_size=params.base_marker_size * text_scale,
            line_width=params.base_line_width * text_scale,
        )
        return scale

    @staticmethod
    def _compute_scale_factor(
        dimension: int, base_scale: float, grid_cells_per_base_size: float
    ) -> float:
        scale_factor = max(base_scale, dimension / grid_cells_per_base_size)
        return scale_factor
