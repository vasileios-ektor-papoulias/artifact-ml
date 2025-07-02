import math
from abc import abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

PlotAutoscaleArgsT = TypeVar("PlotAutoscaleArgsT", bound="PlotAutoscaleArgs")
PlotAutoscaleHyperparamsT = TypeVar("PlotAutoscaleHyperparamsT", bound="PlotAutoscaleHyperparams")
PlotScaleT = TypeVar("PlotScaleT", bound="PlotScale")


@dataclass(frozen=True)
class PlotAutoscaleArgs:
    pass


@dataclass(frozen=True)
class PlotAutoscaleHyperparams:
    min_scale_factor: float
    max_scale_factor: float


@dataclass(frozen=True)
class PlotScale:
    figure_width_scale: float
    figure_height_scale: float


class PlotAutoscaler(Generic[PlotAutoscaleArgsT, PlotAutoscaleHyperparamsT, PlotScaleT]):
    _base_scale: float = 1.0
    _sqrt_threshold: float = 1.0

    @classmethod
    @abstractmethod
    def compute(cls, args: PlotAutoscaleArgsT, params: PlotAutoscaleHyperparamsT) -> PlotScaleT: ...

    @staticmethod
    def _clamp_scale_factor(value: float, min_factor: float, max_factor: float) -> float:
        return max(min_factor, min(max_factor, value))

    @classmethod
    def _compute_text_scale(cls, scale_factor: float, min_factor: float) -> float:
        text_scale = (
            cls._base_scale / math.sqrt(scale_factor)
            if scale_factor > cls._sqrt_threshold
            else cls._base_scale
        )
        text_scale = max(min_factor, text_scale)
        return text_scale
