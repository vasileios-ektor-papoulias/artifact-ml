from enum import Enum
from typing import Callable, Dict, Hashable, Literal, Mapping

import numpy as np

from artifact_core.libs.implementation.binary_classification.confusion.calculator import (
    ConfusionCalculator,
    ConfusionMatrixCell,
)

ConfusionNormalizationStrategyLiteral = Literal["NONE", "TRUE", "PRED", "ALL"]


class ConfusionNormalizationStrategy(Enum):
    NONE = "NONE"  # raw counts
    TRUE = "TRUE"  # normalize rows (per actual/true class)
    PRED = "PRED"  # normalize columns (per predicted class)
    ALL = "ALL"  # normalize globally (sum = 1.0)


class NormalizedConfusionCalculator(ConfusionCalculator):
    @classmethod
    def compute_normalized_confusion_matrix(
        cls,
        true: Mapping[Hashable, str],
        predicted: Mapping[Hashable, str],
        pos_label: str,
        neg_label: str,
        normalization: ConfusionNormalizationStrategy,
    ) -> np.ndarray:
        arr_cm = cls._compute_normalized_confusion_matrix(
            true=true,
            predicted=predicted,
            pos_label=pos_label,
            neg_label=neg_label,
            normalization=normalization,
        )
        return arr_cm

    @classmethod
    def compute_dict_normalized_confusion_counts(
        cls,
        true: Mapping[Hashable, str],
        predicted: Mapping[Hashable, str],
        pos_label: str,
        neg_label: str,
        normalization: ConfusionNormalizationStrategy,
    ) -> Dict[ConfusionMatrixCell, float]:
        arr_cm = cls._compute_normalized_confusion_matrix(
            true=true,
            predicted=predicted,
            pos_label=pos_label,
            neg_label=neg_label,
            normalization=normalization,
        )
        tp, fp, tn, fn = cls._get_counts_from_matrix(arr_cm=arr_cm)
        dict_counts = cls._format_dict_counts(tp=tp, fp=fp, tn=tn, fn=fn)
        return dict_counts

    @classmethod
    def compute_normalized_confusion_count(
        cls,
        confusion_matrix_cell: ConfusionMatrixCell,
        true: Mapping[Hashable, str],
        predicted: Mapping[Hashable, str],
        pos_label: str,
        neg_label: str,
        normalization: ConfusionNormalizationStrategy,
    ) -> float:
        arr_cm = cls._compute_normalized_confusion_matrix(
            true=true,
            predicted=predicted,
            pos_label=pos_label,
            neg_label=neg_label,
            normalization=normalization,
        )
        tp, fp, tn, fn = cls._get_counts_from_matrix(arr_cm=arr_cm)
        count = cls._get_confusion_matrix_cell(
            confusion_matrix_cell=confusion_matrix_cell, tp=tp, fp=fp, tn=tn, fn=fn
        )
        return count

    @classmethod
    def _compute_normalized_confusion_matrix(
        cls,
        true: Mapping[Hashable, str],
        predicted: Mapping[Hashable, str],
        pos_label: str,
        neg_label: str,
        normalization: ConfusionNormalizationStrategy,
    ) -> np.ndarray:
        arr_cm = cls._compute_confusion_matrix(
            true=true, predicted=predicted, pos_label=pos_label, neg_label=neg_label
        )
        arr_cm_normalized = cls._normalize_cm(arr_cm=arr_cm, normalization=normalization)
        return arr_cm_normalized

    @classmethod
    def _normalize_cm(
        cls, arr_cm: np.ndarray, normalization: ConfusionNormalizationStrategy
    ) -> np.ndarray:
        normalizers: Dict[ConfusionNormalizationStrategy, Callable[[np.ndarray], np.ndarray]] = {
            ConfusionNormalizationStrategy.NONE: cls._norm_none,
            ConfusionNormalizationStrategy.TRUE: cls._norm_true,
            ConfusionNormalizationStrategy.PRED: cls._norm_pred,
            ConfusionNormalizationStrategy.ALL: cls._norm_all,
        }
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
