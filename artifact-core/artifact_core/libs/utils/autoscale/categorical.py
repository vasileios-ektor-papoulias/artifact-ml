import math
from dataclasses import dataclass

from artifact_core.libs.utils.autoscale.base import (
    PlotAutoscaleArgs,
    PlotAutoscaleHyperparams,
    PlotAutoscaler,
    PlotScale,
)


@dataclass(frozen=True)
class CategoricalAutoscalerArgs(PlotAutoscaleArgs):
    num_categories: int


@dataclass(frozen=True)
class CategoricalAutoscalerHyperparams(PlotAutoscaleHyperparams):
    min_scale_factor: float
    max_scale_factor: float
    categories_per_base_width: int
    base_figure_width: float
    base_figure_height: float
    base_font_size: float
    base_title_font_size: float
    base_tick_font_size: float
    base_legend_font_size: float
    base_marker_size: float
    base_line_width: float


@dataclass(frozen=True)
class CategoricalPlotScale(PlotScale):
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


class CategoricalAutoscaler(
    PlotAutoscaler[
        CategoricalAutoscalerArgs, CategoricalAutoscalerHyperparams, CategoricalPlotScale
    ]
):
    _height_scale = 1.0

    @classmethod
    def compute(
        cls,
        args: CategoricalAutoscalerArgs,
        params: CategoricalAutoscalerHyperparams,
    ) -> CategoricalPlotScale:
        scale_factor = math.sqrt(args.num_categories / params.categories_per_base_width)
        scale_factor = cls._clamp_scale_factor(
            scale_factor, params.min_scale_factor, params.max_scale_factor
        )
        figure_width_scale = max(1.0, args.num_categories / params.categories_per_base_width)
        figure_width_scale = min(params.max_scale_factor, figure_width_scale)
        text_scale = max(params.min_scale_factor, 1.0 / scale_factor)
        scale = CategoricalPlotScale(
            figure_width_scale=figure_width_scale,
            figure_height_scale=cls._height_scale,
            figure_width=figure_width_scale * params.base_figure_width,
            figure_height=cls._height_scale * params.base_figure_height,
            font_size=params.base_font_size * text_scale,
            title_font_size=params.base_title_font_size * text_scale,
            tick_font_size=params.base_tick_font_size * text_scale,
            legend_font_size=params.base_legend_font_size * text_scale,
            annotation_font_size=params.base_font_size * text_scale,
            marker_size=params.base_marker_size * text_scale,
            line_width=params.base_line_width * text_scale,
        )
        return scale
