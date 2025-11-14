from enum import Enum
from typing import Hashable, Literal, Mapping, Sequence

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

from artifact_core._libs.artifacts.binary_classification.confusion.raw import (
    ConfusionMatrixCell,
    RawConfusionCalculator,
)
from artifact_core._libs.tools.calculators.safe_div_calculator import SafeDivCalculator
from artifact_core._utils.collections.map_aligner import MapAligner

BinaryPredictionMetricLiteral = Literal[
    "ACCURACY",
    "BALANCED_ACCURACY",
    "PRECISION",
    "NPV",
    "RECALL",
    "TNR",
    "FPR",
    "FNR",
    "F1",
    "MCC",
]


class BinaryPredictionMetric(Enum):
    ACCURACY = "accuracy"
    BALANCED_ACCURACY = "balanced_accuracy"
    PRECISION = "precision"
    NPV = "npv"
    RECALL = "recall"
    TNR = "tnr"
    FPR = "fp"
    FNR = "fnr"
    F1 = "f1"
    MCC = "mcc"


class BinaryPredictionMetricCalculator:
    _average: Literal["binary"] = "binary"
    _zero_division: Literal["warn"] = "warn"

    @classmethod
    def compute(
        cls,
        metric_type: BinaryPredictionMetric,
        true: Mapping[Hashable, bool],
        predicted: Mapping[Hashable, bool],
    ) -> float:
        if metric_type is BinaryPredictionMetric.ACCURACY:
            return cls._compute_accuracy(true=true, predicted=predicted)
        elif metric_type is BinaryPredictionMetric.BALANCED_ACCURACY:
            return cls._compute_balanced_accuracy(true=true, predicted=predicted)
        elif metric_type is BinaryPredictionMetric.PRECISION:
            return cls._compute_precision(true=true, predicted=predicted)
        elif metric_type is BinaryPredictionMetric.NPV:
            return cls._compute_npv(true=true, predicted=predicted)
        elif metric_type is BinaryPredictionMetric.RECALL:
            return cls._compute_recall(true=true, predicted=predicted)
        elif metric_type is BinaryPredictionMetric.TNR:
            return cls._compute_tnr(true=true, predicted=predicted)
        elif metric_type is BinaryPredictionMetric.FPR:
            return cls._compute_fpr(true=true, predicted=predicted)
        elif metric_type is BinaryPredictionMetric.FNR:
            return cls._compute_fnr(true=true, predicted=predicted)
        elif metric_type is BinaryPredictionMetric.F1:
            return cls._compute_f1(true=true, predicted=predicted)
        elif metric_type is BinaryPredictionMetric.MCC:
            return cls._compute_mcc(true=true, predicted=predicted)
        else:
            raise ValueError(f"Unsupported classification metric: {metric_type}")

    @classmethod
    def compute_multiple(
        cls,
        metric_types: Sequence[BinaryPredictionMetric],
        true: Mapping[Hashable, bool],
        predicted: Mapping[Hashable, bool],
    ) -> Mapping[BinaryPredictionMetric, float]:
        dict_scores = {
            metric_type: cls.compute(
                metric_type=metric_type,
                true=true,
                predicted=predicted,
            )
            for metric_type in metric_types
        }
        return dict_scores

    @classmethod
    def _compute_accuracy(
        cls,
        true: Mapping[Hashable, bool],
        predicted: Mapping[Hashable, bool],
    ) -> float:
        _, y_true, y_pred = MapAligner.align(left=true, right=predicted)
        score = accuracy_score(y_true=y_true, y_pred=y_pred)
        return float(score)

    @classmethod
    def _compute_balanced_accuracy(
        cls,
        true: Mapping[Hashable, bool],
        predicted: Mapping[Hashable, bool],
    ) -> float:
        _, y_true, y_pred = MapAligner.align(left=true, right=predicted)
        score = balanced_accuracy_score(y_true=y_true, y_pred=y_pred, adjusted=False)
        return float(score)

    @classmethod
    def _compute_precision(
        cls,
        true: Mapping[Hashable, bool],
        predicted: Mapping[Hashable, bool],
    ) -> float:
        _, y_true, y_pred = MapAligner.align(left=true, right=predicted)
        score = precision_score(
            y_true=y_true,
            y_pred=y_pred,
            pos_label=True,
            average=cls._average,
            zero_division=cls._zero_division,
        )
        return float(score)

    @classmethod
    def _compute_recall(
        cls,
        true: Mapping[Hashable, bool],
        predicted: Mapping[Hashable, bool],
    ) -> float:
        _, y_true, y_pred = MapAligner.align(left=true, right=predicted)
        score = recall_score(
            y_true=y_true,
            y_pred=y_pred,
            pos_label=True,
            average=cls._average,
            zero_division=cls._zero_division,
        )
        return float(score)

    @classmethod
    def _compute_f1(
        cls,
        true: Mapping[Hashable, bool],
        predicted: Mapping[Hashable, bool],
    ) -> float:
        _, y_true, y_pred = MapAligner.align(left=true, right=predicted)
        score = f1_score(
            y_true=y_true,
            y_pred=y_pred,
            pos_label=True,
            average=cls._average,
            zero_division=cls._zero_division,
        )
        return float(score)

    @classmethod
    def _compute_mcc(
        cls,
        true: Mapping[Hashable, bool],
        predicted: Mapping[Hashable, bool],
    ) -> float:
        _, y_true, y_pred = MapAligner.align(left=true, right=predicted)
        score = matthews_corrcoef(y_true=y_true, y_pred=y_pred)
        return float(score)

    @classmethod
    def _compute_npv(
        cls,
        true: Mapping[Hashable, bool],
        predicted: Mapping[Hashable, bool],
    ) -> float:
        dict_conf = RawConfusionCalculator.compute_dict_confusion_counts(
            true=true, predicted=predicted
        )
        tn = dict_conf[ConfusionMatrixCell.TRUE_NEGATIVE]
        fn = dict_conf[ConfusionMatrixCell.FALSE_NEGATIVE]
        score = SafeDivCalculator.compute(num=tn, denom=tn + fn)
        return score

    @classmethod
    def _compute_tnr(
        cls,
        true: Mapping[Hashable, bool],
        predicted: Mapping[Hashable, bool],
    ) -> float:
        dict_conf = RawConfusionCalculator.compute_dict_confusion_counts(
            true=true, predicted=predicted
        )
        tn = dict_conf[ConfusionMatrixCell.TRUE_NEGATIVE]
        fp = dict_conf[ConfusionMatrixCell.FALSE_POSITIVE]
        score = SafeDivCalculator.compute(num=tn, denom=tn + fp)
        return score

    @classmethod
    def _compute_fpr(
        cls,
        true: Mapping[Hashable, bool],
        predicted: Mapping[Hashable, bool],
    ) -> float:
        dict_conf = RawConfusionCalculator.compute_dict_confusion_counts(
            true=true, predicted=predicted
        )
        fp = dict_conf[ConfusionMatrixCell.FALSE_POSITIVE]
        tn = dict_conf[ConfusionMatrixCell.TRUE_NEGATIVE]
        score = SafeDivCalculator.compute(num=fp, denom=fp + tn)
        return score

    @classmethod
    def _compute_fnr(
        cls,
        true: Mapping[Hashable, bool],
        predicted: Mapping[Hashable, bool],
    ) -> float:
        dict_conf = RawConfusionCalculator.compute_dict_confusion_counts(
            true=true, predicted=predicted
        )
        fn = dict_conf[ConfusionMatrixCell.FALSE_NEGATIVE]
        tp = dict_conf[ConfusionMatrixCell.TRUE_POSITIVE]
        score = SafeDivCalculator.compute(num=fn, denom=fn + tp)
        return score
