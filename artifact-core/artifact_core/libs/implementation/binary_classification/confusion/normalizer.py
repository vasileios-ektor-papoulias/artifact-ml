from enum import Enum
from typing import Callable, Dict, Literal

import numpy as np

ConfusionNormalizationStrategyLiteral = Literal["NONE", "TRUE", "PRED", "ALL"]


class ConfusionMatrixNormalizationStrategy(Enum):
    NONE = "NONE"  # raw counts
    TRUE = "TRUE"  # normalize rows (per actual/true class)
    PRED = "PRED"  # normalize columns (per predicted class)
    ALL = "ALL"  # normalize globally (sum = 1.0)


class ConfusionMatrixNormalizer:
    @classmethod
    def normalize_cm(
        cls, arr_cm: np.ndarray, normalization: ConfusionMatrixNormalizationStrategy
    ) -> np.ndarray:
        normalizers: Dict[
            ConfusionMatrixNormalizationStrategy, Callable[[np.ndarray], np.ndarray]
        ] = {
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
    def _norm_none(arr_cm: np.ndarray) -> np.ndarray:
        return arr_cm

    @staticmethod
    def _norm_true(arr_cm: np.ndarray) -> np.ndarray:
        row_sums = arr_cm.sum(axis=1, keepdims=True)
        np.divide(arr_cm, row_sums, out=arr_cm, where=row_sums != 0)
        return arr_cm

    @staticmethod
    def _norm_pred(arr_cm: np.ndarray) -> np.ndarray:
        col_sums = arr_cm.sum(axis=0, keepdims=True)
        np.divide(arr_cm, col_sums, out=arr_cm, where=col_sums != 0)
        return arr_cm

    @staticmethod
    def _norm_all(arr_cm: np.ndarray) -> np.ndarray:
        total = arr_cm.sum()
        return arr_cm / total if total != 0 else arr_cm
