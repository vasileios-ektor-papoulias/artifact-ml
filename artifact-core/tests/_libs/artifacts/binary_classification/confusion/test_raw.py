from typing import List

import numpy as np
import pytest
from artifact_core._libs.artifacts.binary_classification.confusion.raw import (
    ConfusionMatrixCell,
    RawConfusionCalculator,
)

from tests._libs.artifacts.binary_classification.conftest import BinaryDataTuple


@pytest.mark.unit
@pytest.mark.parametrize(
    "binary_data_dispatcher, expected_matrix",
    [
        ("binary_data_perfect_small", [[1, 0], [0, 1]]),
        ("binary_data_mixed_equal", [[1, 1], [1, 1]]),
        ("binary_data_all_fn", [[0, 2], [0, 0]]),
        ("binary_data_all_fp", [[0, 0], [2, 0]]),
        ("binary_data_perfect", [[2, 0], [0, 2]]),
    ],
    indirect=["binary_data_dispatcher"],
)
def test_compute_confusion_matrix(
    binary_data_dispatcher: BinaryDataTuple,
    expected_matrix: List[List[int]],
):
    id_to_is_pos, id_to_pred_pos, _ = binary_data_dispatcher
    result = RawConfusionCalculator.compute_confusion_matrix(
        true=id_to_is_pos, predicted=id_to_pred_pos
    )
    np.testing.assert_array_equal(result, expected_matrix)


@pytest.mark.unit
@pytest.mark.parametrize(
    "binary_data_dispatcher, expected_tp, expected_fp, expected_tn, expected_fn",
    [
        ("binary_data_perfect_small", 1, 0, 1, 0),
        ("binary_data_mixed_equal", 1, 1, 1, 1),
        ("binary_data_all_fn", 0, 0, 0, 2),
        ("binary_data_all_fp", 0, 2, 0, 0),
    ],
    indirect=["binary_data_dispatcher"],
)
def test_compute_dict_confusion_counts(
    binary_data_dispatcher: BinaryDataTuple,
    expected_tp: int,
    expected_fp: int,
    expected_tn: int,
    expected_fn: int,
):
    id_to_is_pos, id_to_pred_pos, _ = binary_data_dispatcher
    result = RawConfusionCalculator.compute_dict_confusion_counts(
        true=id_to_is_pos, predicted=id_to_pred_pos
    )
    assert result[ConfusionMatrixCell.TRUE_POSITIVE] == expected_tp
    assert result[ConfusionMatrixCell.FALSE_POSITIVE] == expected_fp
    assert result[ConfusionMatrixCell.TRUE_NEGATIVE] == expected_tn
    assert result[ConfusionMatrixCell.FALSE_NEGATIVE] == expected_fn


@pytest.mark.unit
@pytest.mark.parametrize(
    "binary_data_dispatcher, cell, expected",
    [
        ("binary_data_perfect_small", ConfusionMatrixCell.TRUE_POSITIVE, 1),
        ("binary_data_perfect_small", ConfusionMatrixCell.TRUE_NEGATIVE, 1),
        ("binary_data_perfect_small", ConfusionMatrixCell.FALSE_POSITIVE, 0),
        ("binary_data_perfect_small", ConfusionMatrixCell.FALSE_NEGATIVE, 0),
        ("binary_data_all_fn", ConfusionMatrixCell.FALSE_NEGATIVE, 2),
        ("binary_data_all_fp", ConfusionMatrixCell.FALSE_POSITIVE, 2),
        ("binary_data_mixed_equal", ConfusionMatrixCell.TRUE_POSITIVE, 1),
        ("binary_data_mixed_equal", ConfusionMatrixCell.FALSE_POSITIVE, 1),
        ("binary_data_mixed_equal", ConfusionMatrixCell.TRUE_NEGATIVE, 1),
        ("binary_data_mixed_equal", ConfusionMatrixCell.FALSE_NEGATIVE, 1),
    ],
    indirect=["binary_data_dispatcher"],
)
def test_compute_confusion_count(
    binary_data_dispatcher: BinaryDataTuple,
    cell: ConfusionMatrixCell,
    expected: int,
):
    id_to_is_pos, id_to_pred_pos, _ = binary_data_dispatcher
    result = RawConfusionCalculator.compute_confusion_count(
        confusion_matrix_cell=cell, true=id_to_is_pos, predicted=id_to_pred_pos
    )
    assert result == expected
