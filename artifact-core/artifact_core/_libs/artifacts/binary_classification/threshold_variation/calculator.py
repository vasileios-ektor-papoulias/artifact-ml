from enum import Enum
from typing import Dict, Hashable, Literal, Mapping, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from artifact_core._base.typing.artifact_result import Array
from artifact_core._utils.collections.map_aligner import MapAligner

ThresholdVariationMetricLiteral = Literal["ROC_AUC", "PR_AUC"]


class ThresholdVariationMetric(Enum):
    ROC_AUC = "ROC_AUC"
    PR_AUC = "PR_AUC"


class ThresholdVariationMetricCalculator:
    @classmethod
    def compute(
        cls,
        metric_type: ThresholdVariationMetric,
        true: Mapping[Hashable, bool],
        probs: Mapping[Hashable, float],
    ) -> float:
        if metric_type is ThresholdVariationMetric.ROC_AUC:
            return cls._compute_roc_auc(true=true, probs=probs)
        elif metric_type is ThresholdVariationMetric.PR_AUC:
            return cls._compute_pr_auc(true=true, probs=probs)
        else:
            raise ValueError(f"Unsupported AUC type: {metric_type}")

    @classmethod
    def compute_multiple(
        cls,
        metric_types: Sequence[ThresholdVariationMetric],
        true: Mapping[Hashable, bool],
        probs: Mapping[Hashable, float],
    ) -> Dict[ThresholdVariationMetric, float]:
        return {
            metric_type: cls.compute(metric_type=metric_type, true=true, probs=probs)
            for metric_type in metric_types
        }

    @classmethod
    def _compute_roc_auc(
        cls,
        true: Mapping[Hashable, bool],
        probs: Mapping[Hashable, float],
    ) -> float:
        _, y_true, y_prob = MapAligner.align(left=true, right=probs)
        y_pos = np.asarray(y_true, dtype=int)
        if not cls._has_both_classes(y_pos=y_pos):
            return np.nan
        arr_prob = np.asarray(y_prob, dtype=float)
        score = roc_auc_score(y_true=y_pos, y_score=arr_prob)
        return float(score)

    @classmethod
    def _compute_pr_auc(
        cls,
        true: Mapping[Hashable, bool],
        probs: Mapping[Hashable, float],
    ) -> float:
        _, y_true, y_prob = MapAligner.align(left=true, right=probs)
        y_pos = np.asarray(y_true, dtype=int)
        if not np.any(y_pos == 1):
            return np.nan
        arr_prob = np.asarray(y_prob, dtype=float)
        score = average_precision_score(y_true=y_pos, y_score=arr_prob)
        return float(score)

    @classmethod
    def _has_both_classes(cls, y_pos: Array) -> bool:
        has_pos = np.any(y_pos == 1)
        has_neg = np.any(y_pos == 0)
        has_both = bool(has_pos and has_neg)
        return has_both
