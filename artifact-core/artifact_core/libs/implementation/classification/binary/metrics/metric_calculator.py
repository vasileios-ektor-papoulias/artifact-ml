from enum import Enum
from typing import Dict, Hashable, Iterable, List, Literal, Mapping, Tuple

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

from artifact_core.libs.implementation.classification.binary.confusion.calculator import (
    ConfusionCalculator,
    ConfusionCell,
)


class BinaryClassificationMetric(Enum):
    ACCURACY = "ACCURACY"
    BALANCED_ACCURACY = "BALANCED_ACCURACY"
    PRECISION = "PRECISION"
    NPV = "NPV"
    RECALL = "RECALL"
    TNR = "TNR"
    FPR = "FPR"
    FNR = "FNR"
    F1 = "F1"
    MCC = "MCC"


class BinaryClassificationMetricCalculator:
    _average: Literal["binary"] = "binary"
    _zero_division: Literal["warn"] = "warn"

    @classmethod
    def compute(
        cls,
        metric: BinaryClassificationMetric,
        true: Mapping[Hashable, str],
        predicted: Mapping[Hashable, str],
        pos_label: str,
        neg_label: str,
    ) -> float:
        if metric is BinaryClassificationMetric.ACCURACY:
            return cls._compute_accuracy(true=true, predicted=predicted)
        elif metric is BinaryClassificationMetric.BALANCED_ACCURACY:
            return cls._compute_balanced_accuracy(true=true, predicted=predicted)
        elif metric is BinaryClassificationMetric.PRECISION:
            return cls._compute_precision(true=true, predicted=predicted, pos_label=pos_label)
        elif metric is BinaryClassificationMetric.NPV:
            return cls._compute_npv(
                true=true, predicted=predicted, pos_label=pos_label, neg_label=neg_label
            )
        elif metric is BinaryClassificationMetric.RECALL:
            return cls._compute_recall(true=true, predicted=predicted, pos_label=pos_label)
        elif metric is BinaryClassificationMetric.TNR:
            return cls._compute_tnr(
                true=true, predicted=predicted, pos_label=pos_label, neg_label=neg_label
            )
        elif metric is BinaryClassificationMetric.FPR:
            return cls._compute_fpr(
                true=true, predicted=predicted, pos_label=pos_label, neg_label=neg_label
            )
        elif metric is BinaryClassificationMetric.FNR:
            return cls._compute_fnr(
                true=true, predicted=predicted, pos_label=pos_label, neg_label=neg_label
            )
        elif metric is BinaryClassificationMetric.F1:
            return cls._compute_f1(true=true, predicted=predicted, pos_label=pos_label)
        elif metric is BinaryClassificationMetric.MCC:
            return cls._compute_mcc(true=true, predicted=predicted, pos_label=pos_label)
        else:
            raise ValueError(f"Unsupported classification metric: {metric}")

    @classmethod
    def compute_multiple(
        cls,
        metrics: Iterable[BinaryClassificationMetric],
        true: Mapping[Hashable, str],
        predicted: Mapping[Hashable, str],
        pos_label: str,
        neg_label: str,
    ) -> Dict[BinaryClassificationMetric, float]:
        dict_scores: Dict[BinaryClassificationMetric, float] = {}
        for metric in metrics:
            dict_scores[metric] = cls.compute(
                metric=metric,
                true=true,
                predicted=predicted,
                pos_label=pos_label,
                neg_label=neg_label,
            )
        return dict_scores

    @classmethod
    def _compute_accuracy(
        cls,
        true: Mapping[Hashable, str],
        predicted: Mapping[Hashable, str],
    ) -> float:
        y_true, y_pred = cls._align_labels(true=true, predicted=predicted)
        score = accuracy_score(y_true=y_true, y_pred=y_pred)
        return float(score)

    @classmethod
    def _compute_balanced_accuracy(
        cls, true: Mapping[Hashable, str], predicted: Mapping[Hashable, str]
    ) -> float:
        y_true, y_pred = cls._align_labels(true=true, predicted=predicted)
        score = balanced_accuracy_score(y_true=y_true, y_pred=y_pred, adjusted=False)
        return float(score)

    @classmethod
    def _compute_precision(
        cls,
        true: Mapping[Hashable, str],
        predicted: Mapping[Hashable, str],
        pos_label: str,
    ) -> float:
        y_true, y_pred = cls._align_labels(true=true, predicted=predicted)
        cls._validate_pos_label_present(y_true=y_true, pos_label=pos_label)
        score = precision_score(
            y_true,
            y_pred,
            pos_label=pos_label,
            average=cls._average,
            zero_division=cls._zero_division,
        )
        return float(score)

    @classmethod
    def _compute_npv(
        cls,
        true: Mapping[Hashable, str],
        predicted: Mapping[Hashable, str],
        pos_label: str,
        neg_label: str,
    ) -> float:
        dict_confusion_counts = ConfusionCalculator.compute_confusion_counts(
            true=true, predicted=predicted, pos_label=pos_label, neg_label=neg_label
        )
        tn = dict_confusion_counts[ConfusionCell.TRUE_NEGATIVE]
        fn = dict_confusion_counts[ConfusionCell.FALSE_NEGATIVE]
        score = cls._safe_div(num=tn, denom=tn + fn)
        return score

    @classmethod
    def _compute_recall(
        cls,
        true: Mapping[Hashable, str],
        predicted: Mapping[Hashable, str],
        pos_label: str,
    ) -> float:
        y_true, y_pred = cls._align_labels(true=true, predicted=predicted)
        cls._validate_pos_label_present(y_true=y_true, pos_label=pos_label)
        score = recall_score(
            y_true,
            y_pred,
            pos_label=pos_label,
            average=cls._average,
            zero_division=cls._zero_division,
        )
        return float(score)

    @classmethod
    def _compute_tnr(
        cls,
        true: Mapping[Hashable, str],
        predicted: Mapping[Hashable, str],
        pos_label: str,
        neg_label: str,
    ) -> float:
        dict_confusion_counts = ConfusionCalculator.compute_confusion_counts(
            true=true, predicted=predicted, pos_label=pos_label, neg_label=neg_label
        )
        tn = dict_confusion_counts[ConfusionCell.TRUE_NEGATIVE]
        fp = dict_confusion_counts[ConfusionCell.FALSE_POSITIVE]
        score = cls._safe_div(num=tn, denom=tn + fp)
        return score

    @classmethod
    def _compute_fpr(
        cls,
        true: Mapping[Hashable, str],
        predicted: Mapping[Hashable, str],
        pos_label: str,
        neg_label: str,
    ) -> float:
        dict_confusion_counts = ConfusionCalculator.compute_confusion_counts(
            true=true, predicted=predicted, pos_label=pos_label, neg_label=neg_label
        )
        fp = dict_confusion_counts[ConfusionCell.FALSE_POSITIVE]
        tn = dict_confusion_counts[ConfusionCell.TRUE_NEGATIVE]
        score = cls._safe_div(num=fp, denom=fp + tn)
        return score

    @classmethod
    def _compute_fnr(
        cls,
        true: Mapping[Hashable, str],
        predicted: Mapping[Hashable, str],
        pos_label: str,
        neg_label: str,
    ) -> float:
        dict_confusion_counts = ConfusionCalculator.compute_confusion_counts(
            true=true, predicted=predicted, pos_label=pos_label, neg_label=neg_label
        )
        fn = dict_confusion_counts[ConfusionCell.FALSE_NEGATIVE]
        tp = dict_confusion_counts[ConfusionCell.TRUE_POSITIVE]
        score = cls._safe_div(num=fn, denom=fn + tp)
        return score

    @classmethod
    def _compute_f1(
        cls,
        true: Mapping[Hashable, str],
        predicted: Mapping[Hashable, str],
        pos_label: str,
    ) -> float:
        y_true, y_pred = cls._align_labels(true=true, predicted=predicted)
        cls._validate_pos_label_present(y_true=y_true, pos_label=pos_label)
        score = f1_score(
            y_true,
            y_pred,
            pos_label=pos_label,
            average=cls._average,
            zero_division=cls._zero_division,
        )
        return float(score)

    @classmethod
    def _compute_mcc(
        cls, true: Mapping[Hashable, str], predicted: Mapping[Hashable, str], pos_label: str
    ) -> float:
        y_true, y_pred = cls._align_labels(true=true, predicted=predicted)
        y_true_bin = [1 if label == pos_label else 0 for label in y_true]
        y_pred_bin = [1 if label == pos_label else 0 for label in y_pred]
        score = matthews_corrcoef(y_true_bin, y_pred_bin)
        return float(score)

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
    def _validate_pos_label_present(y_true: List[str], pos_label: str) -> None:
        if pos_label not in set(y_true):
            raise ValueError(
                f"pos_label='{pos_label}' not present in y_true; binary metrics may be undefined."
            )

    @classmethod
    def _safe_div(cls, num: int, denom: int) -> float:
        return 0.0 if denom == 0 else float(num) / float(denom)
