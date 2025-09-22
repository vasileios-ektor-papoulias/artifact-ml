from enum import Enum
from typing import Dict, Hashable, List, Literal, Mapping, Sequence, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ThresholdVariationMetricLiteral = Literal["ROC_AUC", "PR_AUC"]


class ThresholdVariationMetric(Enum):
    ROC_AUC = "ROC_AUC"
    PR_AUC = "PR_AUC"


class ThresholdVariationMetricCalculator:
    @classmethod
    def compute(
        cls,
        metric_type: ThresholdVariationMetric,
        true: Mapping[Hashable, str],
        probs: Mapping[Hashable, float],
        pos_label: str,
    ) -> float:
        if metric_type is ThresholdVariationMetric.ROC_AUC:
            return cls._compute_roc_auc(true=true, probs=probs, pos_label=pos_label)
        elif metric_type is ThresholdVariationMetric.PR_AUC:
            return cls._compute_pr_auc(true=true, probs=probs, pos_label=pos_label)
        else:
            raise ValueError(f"Unsupported AUC type: {metric_type}")

    @classmethod
    def compute_multiple(
        cls,
        metric_types: Sequence[ThresholdVariationMetric],
        true: Mapping[Hashable, str],
        probs: Mapping[Hashable, float],
        pos_label: str,
    ) -> Dict[ThresholdVariationMetric, float]:
        return {
            metric_type: cls.compute(
                metric_type=metric_type, true=true, probs=probs, pos_label=pos_label
            )
            for metric_type in metric_types
        }

    @classmethod
    def _compute_roc_auc(
        cls,
        true: Mapping[Hashable, str],
        probs: Mapping[Hashable, float],
        pos_label: str,
    ) -> float:
        y_true, y_prob = cls._align_labels(true=true, probs=probs)
        y_pos = (np.array(y_true) == pos_label).astype(int)
        has_both_classes = cls._has_both_classes(y_pos=y_pos)
        if not has_both_classes:
            return np.nan
        arr_prob = np.array(y_prob, dtype=float)
        score = roc_auc_score(y_true=y_pos, y_score=arr_prob)
        return float(score)

    @classmethod
    def _compute_pr_auc(
        cls,
        true: Mapping[Hashable, str],
        probs: Mapping[Hashable, float],
        pos_label: str,
    ) -> float:
        y_true, y_prob = cls._align_labels(true=true, probs=probs)
        y_pos = (np.array(y_true) == pos_label).astype(int)
        arr_prob = np.array(y_prob, dtype=float)
        score = average_precision_score(y_true=y_pos, y_score=arr_prob)
        return float(score)

    @classmethod
    def _align_labels(
        cls,
        true: Mapping[Hashable, str],
        probs: Mapping[Hashable, float],
    ) -> Tuple[List[str], List[float]]:
        missing = [k for k in true if k not in probs]
        if missing:
            raise KeyError(
                f"Probabilities missing for {len(missing)} id(s): "
                f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
            )
        keys = list(true.keys())
        y_true = [true[k] for k in keys]
        y_prob = [float(probs[k]) for k in keys]
        return y_true, y_prob

    @classmethod
    def _has_both_classes(cls, y_pos: np.ndarray) -> bool:
        has_pos = np.any(y_pos == 1)
        has_neg = np.any(y_pos == 0)
        has_both = bool(has_pos and has_neg)
        return has_both
