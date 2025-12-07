from typing import List

import pytest
from artifact_core._libs.artifacts.binary_classification.confusion.raw import (
    RawConfusionCalculator,
)
from artifact_core._libs.artifacts.binary_classification.prediction_metrics.calculator import (
    BinaryPredictionMetric,
    BinaryPredictionMetricCalculator,
)
from artifact_core._libs.tools.calculators.safe_div_calculator import SafeDivCalculator
from pytest_mock import MockerFixture

from tests._libs.artifacts.binary_classification.conftest import BinaryDataTuple


@pytest.mark.unit
@pytest.mark.parametrize(
    "binary_data_dispatcher, metric_type, expected",
    [
        ("binary_data_perfect_small", BinaryPredictionMetric.ACCURACY, 1.0),
        ("binary_data_mixed_equal", BinaryPredictionMetric.ACCURACY, 0.5),
        ("binary_data_perfect_small", BinaryPredictionMetric.PRECISION, 1.0),
        ("binary_data_mixed_equal", BinaryPredictionMetric.PRECISION, 0.5),
        ("binary_data_perfect_small", BinaryPredictionMetric.RECALL, 1.0),
        ("binary_data_mixed_equal", BinaryPredictionMetric.RECALL, 0.5),
        ("binary_data_mixed_equal", BinaryPredictionMetric.F1, 0.5),
        ("binary_data_perfect", BinaryPredictionMetric.F1, 1.0),
    ],
    indirect=["binary_data_dispatcher"],
)
def test_compute_sklearn_metrics(
    binary_data_dispatcher: BinaryDataTuple,
    metric_type: BinaryPredictionMetric,
    expected: float,
):
    id_to_is_pos, id_to_pred_pos, _ = binary_data_dispatcher
    result = BinaryPredictionMetricCalculator.compute(
        metric_type=metric_type, true=id_to_is_pos, predicted=id_to_pred_pos
    )
    assert result == pytest.approx(expected=expected)


@pytest.mark.unit
@pytest.mark.parametrize(
    "binary_data_dispatcher, metric_type, expected",
    [
        ("binary_data_perfect_small", BinaryPredictionMetric.NPV, 1.0),
        ("binary_data_perfect_small", BinaryPredictionMetric.TNR, 1.0),
        ("binary_data_imbalanced", BinaryPredictionMetric.FPR, 0.25),
        ("binary_data_mixed_equal", BinaryPredictionMetric.FNR, 0.5),
    ],
    indirect=["binary_data_dispatcher"],
)
def test_compute_confusion_based_metrics(
    mocker: MockerFixture,
    binary_data_dispatcher: BinaryDataTuple,
    metric_type: BinaryPredictionMetric,
    expected: float,
):
    id_to_is_pos, id_to_pred_pos, _ = binary_data_dispatcher
    spy_confusion = mocker.spy(obj=RawConfusionCalculator, name="compute_dict_confusion_counts")
    spy_safe_div = mocker.spy(obj=SafeDivCalculator, name="compute")
    result = BinaryPredictionMetricCalculator.compute(
        metric_type=metric_type, true=id_to_is_pos, predicted=id_to_pred_pos
    )
    assert result == pytest.approx(expected=expected)
    spy_confusion.assert_called_once()
    spy_safe_div.assert_called_once()


@pytest.mark.unit
@pytest.mark.parametrize(
    "binary_data_dispatcher, metric_types",
    [
        ("binary_data_perfect_small", [BinaryPredictionMetric.ACCURACY]),
        (
            "binary_data_perfect_small",
            [BinaryPredictionMetric.PRECISION, BinaryPredictionMetric.RECALL],
        ),
        ("binary_data_perfect_small", [BinaryPredictionMetric.NPV, BinaryPredictionMetric.TNR]),
        ("binary_data_perfect_small", list(BinaryPredictionMetric)),
    ],
    indirect=["binary_data_dispatcher"],
)
def test_compute_multiple(
    binary_data_dispatcher: BinaryDataTuple, metric_types: List[BinaryPredictionMetric]
):
    id_to_is_pos, id_to_pred_pos, _ = binary_data_dispatcher
    result = BinaryPredictionMetricCalculator.compute_multiple(
        metric_types=metric_types, true=id_to_is_pos, predicted=id_to_pred_pos
    )
    assert set(result.keys()) == set(metric_types)
    for metric, value in result.items():
        assert isinstance(value, float)
