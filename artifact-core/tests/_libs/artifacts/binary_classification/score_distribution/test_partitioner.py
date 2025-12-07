from typing import List

import numpy as np
import pytest
from artifact_core._libs.artifacts.binary_classification.score_distribution.partitioner import (
    BinarySamplePartitioner,
    BinarySampleSplit,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    "y_true_bin, y_prob, split, expected_len",
    [
        ([True, True, False, False], [0.9, 0.8, 0.2, 0.1], BinarySampleSplit.ALL, 4),
        ([True, True, False, False], [0.9, 0.8, 0.2, 0.1], BinarySampleSplit.POSITIVE, 2),
        ([True, True, False, False], [0.9, 0.8, 0.2, 0.1], BinarySampleSplit.NEGATIVE, 2),
        ([True, False], [0.9, 0.1], BinarySampleSplit.POSITIVE, 1),
        ([True, False], [0.9, 0.1], BinarySampleSplit.NEGATIVE, 1),
        ([], [], BinarySampleSplit.ALL, 0),
    ],
)
def test_partition(
    y_true_bin: List[bool],
    y_prob: List[float],
    split: BinarySampleSplit,
    expected_len: int,
):
    result = BinarySamplePartitioner.partition(y_true_bin=y_true_bin, y_prob=y_prob, split=split)
    assert isinstance(result, np.ndarray)
    assert len(result) == expected_len


@pytest.mark.unit
@pytest.mark.parametrize(
    "y_true_bin, y_prob, split, expected_values",
    [
        ([True, True, False, False], [0.9, 0.8, 0.2, 0.1], BinarySampleSplit.POSITIVE, [0.9, 0.8]),
        ([True, True, False, False], [0.9, 0.8, 0.2, 0.1], BinarySampleSplit.NEGATIVE, [0.2, 0.1]),
        ([True, False, True], [0.9, 0.1, 0.7], BinarySampleSplit.ALL, [0.9, 0.1, 0.7]),
    ],
)
def test_partition_values(
    y_true_bin: List[bool],
    y_prob: List[float],
    split: BinarySampleSplit,
    expected_values: List[float],
):
    result = BinarySamplePartitioner.partition(y_true_bin=y_true_bin, y_prob=y_prob, split=split)
    np.testing.assert_array_almost_equal(result, expected_values)
