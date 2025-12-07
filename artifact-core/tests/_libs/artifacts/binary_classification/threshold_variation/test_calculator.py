from typing import List, Mapping

import numpy as np
import pytest
from artifact_core._libs.artifacts.binary_classification.threshold_variation.calculator import (
    ThresholdVariationMetric,
    ThresholdVariationMetricCalculator,
)

from tests._libs.artifacts.binary_classification.conftest import BinaryDataTuple


@pytest.mark.unit
@pytest.mark.parametrize(
    "binary_data_dispatcher, metric_type, expected",
    [
        ("binary_data_perfect", ThresholdVariationMetric.ROC_AUC, 1.0),
        ("binary_data_perfect", ThresholdVariationMetric.PR_AUC, 1.0),
        ("binary_data_perfect_small", ThresholdVariationMetric.ROC_AUC, 1.0),
        ("binary_data_perfect_small", ThresholdVariationMetric.PR_AUC, 1.0),
        ("binary_data_imbalanced", ThresholdVariationMetric.ROC_AUC, 1.0),
        ("binary_data_imbalanced", ThresholdVariationMetric.PR_AUC, 1.0),
        ("binary_data_balanced", ThresholdVariationMetric.ROC_AUC, 8 / 9),
        ("binary_data_balanced", ThresholdVariationMetric.PR_AUC, 11 / 12),
        ("binary_data_mixed_equal", ThresholdVariationMetric.ROC_AUC, 0.75),
        ("binary_data_mixed_equal", ThresholdVariationMetric.PR_AUC, 5 / 6),
    ],
    indirect=["binary_data_dispatcher"],
)
def test_compute(
    binary_data_dispatcher: BinaryDataTuple,
    metric_type: ThresholdVariationMetric,
    expected: float,
):
    id_to_is_pos, _, id_to_prob_pos = binary_data_dispatcher
    result = ThresholdVariationMetricCalculator.compute(
        metric_type=metric_type, true=id_to_is_pos, probs=id_to_prob_pos
    )
    assert result == pytest.approx(expected=expected)


@pytest.mark.unit
@pytest.mark.parametrize(
    "binary_data_dispatcher, metric_types, expected",
    [
        (
            "binary_data_perfect_small",
            [ThresholdVariationMetric.ROC_AUC],
            {ThresholdVariationMetric.ROC_AUC: 1.0},
        ),
        (
            "binary_data_perfect",
            [ThresholdVariationMetric.ROC_AUC, ThresholdVariationMetric.PR_AUC],
            {ThresholdVariationMetric.ROC_AUC: 1.0, ThresholdVariationMetric.PR_AUC: 1.0},
        ),
        (
            "binary_data_balanced",
            [ThresholdVariationMetric.ROC_AUC, ThresholdVariationMetric.PR_AUC],
            {ThresholdVariationMetric.ROC_AUC: 8 / 9, ThresholdVariationMetric.PR_AUC: 11 / 12},
        ),
        (
            "binary_data_mixed_equal",
            [ThresholdVariationMetric.ROC_AUC, ThresholdVariationMetric.PR_AUC],
            {ThresholdVariationMetric.ROC_AUC: 0.75, ThresholdVariationMetric.PR_AUC: 5 / 6},
        ),
    ],
    indirect=["binary_data_dispatcher"],
)
def test_compute_multiple(
    binary_data_dispatcher: BinaryDataTuple,
    metric_types: List[ThresholdVariationMetric],
    expected: Mapping[ThresholdVariationMetric, float],
):
    id_to_is_pos, _, id_to_prob_pos = binary_data_dispatcher
    result = ThresholdVariationMetricCalculator.compute_multiple(
        metric_types=metric_types, true=id_to_is_pos, probs=id_to_prob_pos
    )
    assert set(result.keys()) == set(expected.keys())
    for metric, value in result.items():
        assert value == pytest.approx(expected=expected[metric])


@pytest.mark.unit
@pytest.mark.parametrize(
    "binary_data_dispatcher",
    ["binary_data_all_positive", "binary_data_all_negative"],
    indirect=True,
)
def test_compute_roc_auc_single_class_returns_nan(binary_data_dispatcher: BinaryDataTuple):
    id_to_is_pos, _, id_to_prob_pos = binary_data_dispatcher
    result = ThresholdVariationMetricCalculator.compute(
        metric_type=ThresholdVariationMetric.ROC_AUC, true=id_to_is_pos, probs=id_to_prob_pos
    )
    assert np.isnan(result)


@pytest.mark.unit
@pytest.mark.parametrize("binary_data_dispatcher", ["binary_data_all_negative"], indirect=True)
def test_compute_pr_auc_no_positive_returns_nan(binary_data_dispatcher: BinaryDataTuple):
    id_to_is_pos, _, id_to_prob_pos = binary_data_dispatcher
    result = ThresholdVariationMetricCalculator.compute(
        metric_type=ThresholdVariationMetric.PR_AUC, true=id_to_is_pos, probs=id_to_prob_pos
    )
    assert np.isnan(result)
