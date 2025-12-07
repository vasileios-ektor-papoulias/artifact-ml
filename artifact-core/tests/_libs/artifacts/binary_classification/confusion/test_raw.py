from typing import Dict, Hashable, List

import numpy as np
import pytest
from artifact_core._libs.artifacts.binary_classification.confusion.raw import (
    ConfusionMatrixCell,
    RawConfusionCalculator,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    "true, predicted, expected_matrix",
    [
        (
            {0: True, 1: False},
            {0: True, 1: False},
            [[1, 0], [0, 1]],
        ),
        (
            {0: True, 1: True, 2: False, 3: False},
            {0: True, 1: False, 2: True, 3: False},
            [[1, 1], [1, 1]],
        ),
        (
            {0: True, 1: True},
            {0: False, 1: False},
            [[0, 2], [0, 0]],
        ),
        (
            {0: False, 1: False},
            {0: True, 1: True},
            [[0, 0], [2, 0]],
        ),
        (
            {0: True, 1: True, 2: False, 3: False},
            {0: True, 1: True, 2: False, 3: False},
            [[2, 0], [0, 2]],
        ),
    ],
)
def test_compute_confusion_matrix(
    true: Dict[Hashable, bool],
    predicted: Dict[Hashable, bool],
    expected_matrix: List[List[int]],
):
    result = RawConfusionCalculator.compute_confusion_matrix(true=true, predicted=predicted)
    np.testing.assert_array_equal(result, expected_matrix)


@pytest.mark.unit
@pytest.mark.parametrize(
    "true, predicted, expected_tp, expected_fp, expected_tn, expected_fn",
    [
        ({0: True, 1: False}, {0: True, 1: False}, 1, 0, 1, 0),
        (
            {0: True, 1: True, 2: False, 3: False},
            {0: True, 1: False, 2: True, 3: False},
            1,
            1,
            1,
            1,
        ),
        ({0: True, 1: True}, {0: False, 1: False}, 0, 0, 0, 2),
        ({0: False, 1: False}, {0: True, 1: True}, 0, 2, 0, 0),
    ],
)
def test_compute_dict_confusion_counts(
    true: Dict[Hashable, bool],
    predicted: Dict[Hashable, bool],
    expected_tp: int,
    expected_fp: int,
    expected_tn: int,
    expected_fn: int,
):
    result = RawConfusionCalculator.compute_dict_confusion_counts(true=true, predicted=predicted)
    assert result[ConfusionMatrixCell.TRUE_POSITIVE] == expected_tp
    assert result[ConfusionMatrixCell.FALSE_POSITIVE] == expected_fp
    assert result[ConfusionMatrixCell.TRUE_NEGATIVE] == expected_tn
    assert result[ConfusionMatrixCell.FALSE_NEGATIVE] == expected_fn


@pytest.mark.unit
@pytest.mark.parametrize(
    "true, predicted, cell, expected",
    [
        ({0: True, 1: False}, {0: True, 1: False}, ConfusionMatrixCell.TRUE_POSITIVE, 1),
        ({0: True, 1: False}, {0: True, 1: False}, ConfusionMatrixCell.TRUE_NEGATIVE, 1),
        ({0: True, 1: False}, {0: True, 1: False}, ConfusionMatrixCell.FALSE_POSITIVE, 0),
        ({0: True, 1: False}, {0: True, 1: False}, ConfusionMatrixCell.FALSE_NEGATIVE, 0),
        ({0: True, 1: True}, {0: False, 1: False}, ConfusionMatrixCell.FALSE_NEGATIVE, 2),
        ({0: False, 1: False}, {0: True, 1: True}, ConfusionMatrixCell.FALSE_POSITIVE, 2),
    ],
)
def test_compute_confusion_count(
    true: Dict[Hashable, bool],
    predicted: Dict[Hashable, bool],
    cell: ConfusionMatrixCell,
    expected: int,
):
    result = RawConfusionCalculator.compute_confusion_count(
        confusion_matrix_cell=cell, true=true, predicted=predicted
    )
    assert result == expected
