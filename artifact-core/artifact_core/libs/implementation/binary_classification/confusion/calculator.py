from enum import Enum
from typing import Dict, Hashable, Iterable, List, Mapping, Tuple

import numpy as np
from sklearn.metrics import confusion_matrix


class ConfusionCell(Enum):
    TRUE_POSITIVE = "TRUE_POSITIVE"
    FALSE_POSITIVE = "FALSE_POSITIVE"
    TRUE_NEGATIVE = "TRUE_NEGATIVE"
    FALSE_NEGATIVE = "FALSE_NEGATIVE"


class ConfusionCalculator:
    @classmethod
    def compute_confusion_matrix(
        cls,
        true: Mapping[Hashable, str],
        predicted: Mapping[Hashable, str],
        pos_label: str,
        neg_label: str,
    ) -> np.ndarray:
        arr_cm = cls._compute_confusion_matrix(
            true=true, predicted=predicted, pos_label=pos_label, neg_label=neg_label
        )
        return arr_cm

    @classmethod
    def compute_confusion_counts(
        cls,
        true: Mapping[Hashable, str],
        predicted: Mapping[Hashable, str],
        pos_label: str,
        neg_label: str,
    ) -> Dict[ConfusionCell, int]:
        tp, fp, tn, fn = cls._compute_confusion_counts(
            true=true, predicted=predicted, pos_label=pos_label, neg_label=neg_label
        )
        return {
            ConfusionCell.TRUE_POSITIVE: tp,
            ConfusionCell.FALSE_POSITIVE: fp,
            ConfusionCell.TRUE_NEGATIVE: tn,
            ConfusionCell.FALSE_NEGATIVE: fn,
        }

    @classmethod
    def compute_confusion_cell(
        cls,
        cell: ConfusionCell,
        true: Mapping[Hashable, str],
        predicted: Mapping[Hashable, str],
        pos_label: str,
        neg_label: str,
    ) -> int:
        tp, fp, tn, fn = cls._compute_confusion_counts(
            true=true, predicted=predicted, pos_label=pos_label, neg_label=neg_label
        )
        if cell is ConfusionCell.TRUE_POSITIVE:
            return tp
        elif cell is ConfusionCell.FALSE_POSITIVE:
            return fp
        elif cell is ConfusionCell.TRUE_NEGATIVE:
            return tn
        elif cell is ConfusionCell.FALSE_NEGATIVE:
            return fn
        else:
            raise ValueError(f"Unsupported confusion matrix cell: {cell}")

    @classmethod
    def _compute_confusion_counts(
        cls,
        true: Mapping[Hashable, str],
        predicted: Mapping[Hashable, str],
        pos_label: str,
        neg_label: str,
    ) -> Tuple[int, int, int, int]:
        arr_cm = cls._compute_confusion_matrix(
            true=true, predicted=predicted, pos_label=pos_label, neg_label=neg_label
        )
        tp = int(arr_cm[0, 0])
        fn = int(arr_cm[0, 1])
        fp = int(arr_cm[1, 0])
        tn = int(arr_cm[1, 1])
        return tp, fp, tn, fn

    @classmethod
    def _compute_confusion_matrix(
        cls,
        true: Mapping[Hashable, str],
        predicted: Mapping[Hashable, str],
        pos_label: str,
        neg_label: str,
    ) -> np.ndarray:
        y_true, y_pred = cls._align_labels(true=true, predicted=predicted)
        labels = [pos_label, neg_label]
        cls._assert_known_labels(observed=y_true, known=labels)
        cls._assert_known_labels(observed=y_pred, known=labels)
        arr_cm = confusion_matrix(y_true, y_pred, labels=labels)
        return arr_cm

    @staticmethod
    def _align_labels(
        true: Mapping[Hashable, str],
        predicted: Mapping[Hashable, str],
    ) -> Tuple[List[str], List[str]]:
        missing = [k for k in true.keys() if k not in predicted]
        if missing:
            raise KeyError(
                f"Predictions missing for {len(missing)} id(s): "
                f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
            )
        keys = list(true.keys())
        y_true = [true[k] for k in keys]
        y_pred = [predicted[k] for k in keys]
        return y_true, y_pred

    @staticmethod
    def _assert_known_labels(observed: Iterable[str], known: Iterable[str]) -> None:
        known = set(known)
        unknown = set(observed) - set(known)
        if unknown:
            raise ValueError(
                f"Unexpected labels found in data: {sorted(unknown)}; "
                f"expected only {sorted(known)}."
            )
