from enum import Enum
from typing import Callable, Dict, Literal

import numpy as np

from artifact_core._base.types.artifact_result import Array

ConfusionNormalizationStrategyLiteral = Literal["NONE", "TRUE", "PRED", "ALL"]


class ConfusionMatrixNormalizationStrategy(Enum):
    NONE = "NONE"  # raw counts
    TRUE = "TRUE"  # normalize rows (per actual/true class)
    PRED = "PRED"  # normalize columns (per predicted class)
    ALL = "ALL"  # normalize globally (sum = 1.0)


class ConfusionMatrixNormalizer:
    @classmethod
    def normalize_cm(
        cls, arr_cm: Array, normalization: ConfusionMatrixNormalizationStrategy
    ) -> Array:
        normalizers: Dict[ConfusionMatrixNormalizationStrategy, Callable[[Array], Array]] = {
            ConfusionMatrixNormalizationStrategy.NONE: cls._norm_none,
            ConfusionMatrixNormalizationStrategy.TRUE: cls._norm_true,
            ConfusionMatrixNormalizationStrategy.PRED: cls._norm_pred,
            ConfusionMatrixNormalizationStrategy.ALL: cls._norm_all,
        }
        arr_cm = arr_cm.astype(np.float64, copy=True)
        try:
            return normalizers[normalization](arr_cm.copy())
        except KeyError as e:
            raise ValueError(f"Unknown normalization mode: {normalization}") from e

    @staticmethod
    def _norm_none(arr_cm: Array) -> Array:
        return arr_cm

    @staticmethod
    def _norm_true(arr_cm: Array) -> Array:
        row_sums = arr_cm.sum(axis=1, keepdims=True)
        np.divide(arr_cm, row_sums, out=arr_cm, where=row_sums != 0)
        return arr_cm

    @staticmethod
    def _norm_pred(arr_cm: Array) -> Array:
        col_sums = arr_cm.sum(axis=0, keepdims=True)
        np.divide(arr_cm, col_sums, out=arr_cm, where=col_sums != 0)
        return arr_cm

    @staticmethod
    def _norm_all(arr_cm: Array) -> Array:
        total = arr_cm.sum()
        return arr_cm / total if total != 0 else arr_cm
