import math
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class GridScaleConfig:
    base_figure_width: float = 7.0
    base_figure_height: float = 7.0
    base_font_size: float = 14.0
    base_title_font_size: float = 16.0
    base_tick_font_size: float = 12.0
    base_legend_font_size: float = 12.0
    base_annotation_font_size: float = 8.0
    base_marker_size: float = 5.0
    base_line_width: float = 2.0
    min_scale_factor: float = 0.9
    max_scale_factor: float = 10.0
    grid_cells_per_base_size: int = 3


@dataclass(frozen=True)
class GridScaleResult:
    figure_width_scale: float
    figure_height_scale: float
    font_size: float
    title_font_size: float
    tick_font_size: float
    legend_font_size: float
    annotation_font_size: float
    marker_size: float
    line_width: float


class GridAutoscaler:
    _base_scale = 1.0
    _sqrt_threshold = 1.0

    @classmethod
    def calculate_scale(
        cls, grid_size: Tuple[int, int], config: Optional[GridScaleConfig] = None
    ) -> GridScaleResult:
        if config is None:
            config = GridScaleConfig()
        total_cells = grid_size[0] * grid_size[1]
        grid_dimension = math.sqrt(max(1, total_cells))
        scale_factor = max(cls._base_scale, grid_dimension / config.grid_cells_per_base_size)
        width_scale = scale_factor
        height_scale = scale_factor
        width_scale = min(config.max_scale_factor, width_scale)
        height_scale = min(config.max_scale_factor, height_scale)
        avg_scale = math.sqrt(width_scale * height_scale)
        text_scale = cls._calculate_text_scale(avg_scale, config.min_scale_factor)
        return GridScaleResult(
            figure_width_scale=width_scale,
            figure_height_scale=height_scale,
            font_size=config.base_font_size * text_scale,
            title_font_size=config.base_title_font_size * text_scale,
            tick_font_size=config.base_tick_font_size * text_scale,
            legend_font_size=config.base_legend_font_size * text_scale,
            annotation_font_size=config.base_annotation_font_size * text_scale,
            marker_size=config.base_marker_size * text_scale,
            line_width=config.base_line_width * text_scale,
        )

    @staticmethod
    def _clamp_scale_factor(value: float, min_factor: float, max_factor: float) -> float:
        return max(min_factor, min(max_factor, value))

    @classmethod
    def _calculate_text_scale(cls, scale_factor: float, min_factor: float) -> float:
        text_scale = (
            cls._base_scale / math.sqrt(scale_factor)
            if scale_factor > cls._sqrt_threshold
            else cls._base_scale
        )
        return max(min_factor, text_scale)
