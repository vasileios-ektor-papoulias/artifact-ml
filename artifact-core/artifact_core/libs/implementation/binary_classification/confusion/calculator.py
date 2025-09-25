from enum import Enum
from typing import Dict, Hashable, Mapping, Tuple

import numpy as np
from sklearn.metrics import confusion_matrix

from artifact_core.libs.utils.misc.dict_aligner import DictAligner


class ConfusionMatrixCell(Enum):
    TRUE_POSITIVE = "true_positive"
    FALSE_POSITIVE = "false_positive"
    TRUE_NEGATIVE = "true_negative"
    FALSE_NEGATIVE = "false_negative"


class ConfusionCalculator:
    @classmethod
    def compute_confusion_matrix(
        cls,
        true: Mapping[Hashable, bool],
        predicted: Mapping[Hashable, bool],
    ) -> np.ndarray:
        arr_cm = cls._compute_confusion_matrix(true=true, predicted=predicted)
        return arr_cm

    @classmethod
    def compute_dict_confusion_counts(
        cls,
        true: Mapping[Hashable, bool],
        predicted: Mapping[Hashable, bool],
    ) -> Dict[ConfusionMatrixCell, float]:
        arr_cm = cls._compute_confusion_matrix(true=true, predicted=predicted)
        tp, fp, tn, fn = cls._get_counts_from_matrix(arr_cm=arr_cm)
        dict_counts = cls._format_dict_counts(tp=tp, fp=fp, tn=tn, fn=fn)
        return dict_counts

    @classmethod
    def compute_confusion_count(
        cls,
        confusion_matrix_cell: ConfusionMatrixCell,
        true: Mapping[Hashable, bool],
        predicted: Mapping[Hashable, bool],
    ) -> float:
        arr_cm = cls._compute_confusion_matrix(true=true, predicted=predicted)
        tp, fp, tn, fn = cls._get_counts_from_matrix(arr_cm=arr_cm)
        count = cls._get_confusion_matrix_cell(
            confusion_matrix_cell=confusion_matrix_cell, tp=tp, fp=fp, tn=tn, fn=fn
        )
        return count

    @classmethod
    def _format_dict_counts(
        cls, tp: float, fp: float, tn: float, fn: float
    ) -> Dict[ConfusionMatrixCell, float]:
        return {
            ConfusionMatrixCell.TRUE_POSITIVE: tp,
            ConfusionMatrixCell.FALSE_POSITIVE: fp,
            ConfusionMatrixCell.TRUE_NEGATIVE: tn,
            ConfusionMatrixCell.FALSE_NEGATIVE: fn,
        }

    @classmethod
    def _get_confusion_matrix_cell(
        cls, confusion_matrix_cell: ConfusionMatrixCell, tp: float, fp: float, tn: float, fn: float
    ) -> float:
        if confusion_matrix_cell is ConfusionMatrixCell.TRUE_POSITIVE:
            return tp
        elif confusion_matrix_cell is ConfusionMatrixCell.FALSE_POSITIVE:
            return fp
        elif confusion_matrix_cell is ConfusionMatrixCell.TRUE_NEGATIVE:
            return tn
        elif confusion_matrix_cell is ConfusionMatrixCell.FALSE_NEGATIVE:
            return fn
        else:
            raise ValueError(f"Unsupported confusion matrix cell: {confusion_matrix_cell}")

    @staticmethod
    def _get_counts_from_matrix(arr_cm: np.ndarray) -> Tuple[int, int, int, int]:
        # With labels=[True, False], layout is:
        # [[TP, FN],
        #  [FP, TN]]
        tp = int(arr_cm[0, 0])
        fn = int(arr_cm[0, 1])
        fp = int(arr_cm[1, 0])
        tn = int(arr_cm[1, 1])
        return tp, fp, tn, fn

    @classmethod
    def _compute_confusion_matrix(
        cls,
        true: Mapping[Hashable, bool],
        predicted: Mapping[Hashable, bool],
    ) -> np.ndarray:
        _, y_true, y_pred = DictAligner.align(left=true, right=predicted)
        arr_cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=[True, False])
        return arr_cm
