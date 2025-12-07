from typing import List

import numpy as np
import pytest
from artifact_core._libs.artifacts.binary_classification.confusion.calculator import (
    NormalizedConfusionCalculator,
)
from artifact_core._libs.artifacts.binary_classification.confusion.normalizer import (
    ConfusionMatrixNormalizationStrategy,
    ConfusionMatrixNormalizer,
)
from artifact_core._libs.artifacts.binary_classification.confusion.raw import (
    ConfusionMatrixCell,
    RawConfusionCalculator,
)
from pytest_mock import MockerFixture

from tests._libs.artifacts.binary_classification.conftest import BinaryDataTuple


@pytest.mark.unit
@pytest.mark.parametrize(
    "binary_data_dispatcher, normalization, expected_matrix",
    [
        (
            "binary_data_perfect_small",
            ConfusionMatrixNormalizationStrategy.NONE,
            [[1.0, 0.0], [0.0, 1.0]],
        ),
        (
            "binary_data_perfect_small",
            ConfusionMatrixNormalizationStrategy.TRUE,
            [[1.0, 0.0], [0.0, 1.0]],
        ),
        (
            "binary_data_perfect_small",
            ConfusionMatrixNormalizationStrategy.ALL,
            [[0.5, 0.0], [0.0, 0.5]],
        ),
        (
            "binary_data_mixed_equal",
            ConfusionMatrixNormalizationStrategy.ALL,
            [[0.25, 0.25], [0.25, 0.25]],
        ),
        (
            "binary_data_perfect",
            ConfusionMatrixNormalizationStrategy.TRUE,
            [[1.0, 0.0], [0.0, 1.0]],
        ),
    ],
    indirect=["binary_data_dispatcher"],
)
def test_compute_normalized_confusion_matrix(
    mocker: MockerFixture,
    binary_data_dispatcher: BinaryDataTuple,
    normalization: ConfusionMatrixNormalizationStrategy,
    expected_matrix: List[List[float]],
):
    id_to_is_pos, id_to_pred_pos, _ = binary_data_dispatcher
    spy_raw = mocker.spy(obj=RawConfusionCalculator, name="_compute_confusion_matrix")
    spy_normalizer = mocker.spy(obj=ConfusionMatrixNormalizer, name="normalize_cm")
    result = NormalizedConfusionCalculator.compute_normalized_confusion_matrix(
        true=id_to_is_pos, predicted=id_to_pred_pos, normalization=normalization
    )
    np.testing.assert_array_almost_equal(result, expected_matrix)
    spy_raw.assert_called_once()
    spy_normalizer.assert_called_once()
    assert spy_normalizer.call_args.kwargs["normalization"] == normalization


@pytest.mark.unit
@pytest.mark.parametrize(
    "binary_data_dispatcher, normalization_types, expected_count",
    [
        ("binary_data_perfect_small", [ConfusionMatrixNormalizationStrategy.TRUE], 1),
        (
            "binary_data_mixed_equal",
            [ConfusionMatrixNormalizationStrategy.TRUE, ConfusionMatrixNormalizationStrategy.PRED],
            2,
        ),
        ("binary_data_perfect_small", list(ConfusionMatrixNormalizationStrategy), 4),
    ],
    indirect=["binary_data_dispatcher"],
)
def test_compute_confusion_matrix_multiple_normalizations(
    mocker: MockerFixture,
    binary_data_dispatcher: BinaryDataTuple,
    normalization_types: List[ConfusionMatrixNormalizationStrategy],
    expected_count: int,
):
    id_to_is_pos, id_to_pred_pos, _ = binary_data_dispatcher
    spy_normalizer = mocker.spy(obj=ConfusionMatrixNormalizer, name="normalize_cm")
    result = NormalizedConfusionCalculator.compute_confusion_matrix_multiple_normalizations(
        true=id_to_is_pos, predicted=id_to_pred_pos, normalization_types=normalization_types
    )
    assert len(result) == expected_count
    assert set(result.keys()) == set(normalization_types)
    assert spy_normalizer.call_count == expected_count


@pytest.mark.unit
@pytest.mark.parametrize(
    "binary_data_dispatcher, cell, normalization, expected",
    [
        (
            "binary_data_perfect_small",
            ConfusionMatrixCell.TRUE_POSITIVE,
            ConfusionMatrixNormalizationStrategy.TRUE,
            1.0,
        ),
        (
            "binary_data_perfect_small",
            ConfusionMatrixCell.TRUE_NEGATIVE,
            ConfusionMatrixNormalizationStrategy.TRUE,
            1.0,
        ),
        (
            "binary_data_perfect_small",
            ConfusionMatrixCell.TRUE_POSITIVE,
            ConfusionMatrixNormalizationStrategy.ALL,
            0.5,
        ),
        (
            "binary_data_mixed_equal",
            ConfusionMatrixCell.FALSE_POSITIVE,
            ConfusionMatrixNormalizationStrategy.ALL,
            0.25,
        ),
        (
            "binary_data_mixed_equal",
            ConfusionMatrixCell.TRUE_POSITIVE,
            ConfusionMatrixNormalizationStrategy.ALL,
            0.25,
        ),
    ],
    indirect=["binary_data_dispatcher"],
)
def test_compute_normalized_confusion_count(
    mocker: MockerFixture,
    binary_data_dispatcher: BinaryDataTuple,
    cell: ConfusionMatrixCell,
    normalization: ConfusionMatrixNormalizationStrategy,
    expected: float,
):
    id_to_is_pos, id_to_pred_pos, _ = binary_data_dispatcher
    spy_normalizer = mocker.spy(obj=ConfusionMatrixNormalizer, name="normalize_cm")
    result = NormalizedConfusionCalculator.compute_normalized_confusion_count(
        confusion_matrix_cell=cell,
        true=id_to_is_pos,
        predicted=id_to_pred_pos,
        normalization=normalization,
    )
    assert result == pytest.approx(expected=expected)
    spy_normalizer.assert_called_once()


@pytest.mark.unit
@pytest.mark.parametrize(
    "binary_data_dispatcher, normalization, expected_total",
    [
        ("binary_data_perfect_small", ConfusionMatrixNormalizationStrategy.ALL, 1.0),
        ("binary_data_mixed_equal", ConfusionMatrixNormalizationStrategy.ALL, 1.0),
        ("binary_data_perfect", ConfusionMatrixNormalizationStrategy.ALL, 1.0),
    ],
    indirect=["binary_data_dispatcher"],
)
def test_compute_dict_normalized_confusion_counts(
    mocker: MockerFixture,
    binary_data_dispatcher: BinaryDataTuple,
    normalization: ConfusionMatrixNormalizationStrategy,
    expected_total: float,
):
    id_to_is_pos, id_to_pred_pos, _ = binary_data_dispatcher
    spy_normalizer = mocker.spy(obj=ConfusionMatrixNormalizer, name="normalize_cm")
    result = NormalizedConfusionCalculator.compute_dict_normalized_confusion_counts(
        true=id_to_is_pos, predicted=id_to_pred_pos, normalization=normalization
    )
    assert set(result.keys()) == set(ConfusionMatrixCell)
    assert sum(result.values()) == pytest.approx(expected=expected_total)
    spy_normalizer.assert_called_once()
