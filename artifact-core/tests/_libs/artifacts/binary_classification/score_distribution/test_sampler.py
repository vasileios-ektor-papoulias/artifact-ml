from typing import List, Mapping

import pytest
from artifact_core._libs.artifacts.binary_classification.score_distribution.partitioner import (
    BinarySamplePartitioner,
    BinarySampleSplit,
)
from artifact_core._libs.artifacts.binary_classification.score_distribution.sampler import (
    ScoreDistributionSampler,
)
from pytest_mock import MockerFixture

from tests._libs.artifacts.binary_classification.conftest import BinaryDataTuple

# Pre-computed expected samples for binary_data_balanced (aligned by id 0,1,2,3,4,5):
# - y_true_bin = [True, True, False, False, True, False]
# - y_prob = [0.9, 0.4, 0.2, 0.6, 0.8, 0.1]
_BALANCED_SAMPLES_ALL = [0.9, 0.4, 0.2, 0.6, 0.8, 0.1]
_BALANCED_SAMPLES_POS = [0.9, 0.4, 0.8]
_BALANCED_SAMPLES_NEG = [0.2, 0.6, 0.1]

# Pre-computed expected samples for binary_data_imbalanced (aligned by id 0,1,2,3,4):
# - y_true_bin = [True, False, False, False, False]
# - y_prob = [0.95, 0.1, 0.55, 0.2, 0.05]
_IMBALANCED_SAMPLES_POS = [0.95]


@pytest.mark.unit
@pytest.mark.parametrize(
    "binary_data_dispatcher, split, expected",
    [
        ("binary_data_balanced", BinarySampleSplit.ALL, _BALANCED_SAMPLES_ALL),
        ("binary_data_balanced", BinarySampleSplit.POSITIVE, _BALANCED_SAMPLES_POS),
        ("binary_data_balanced", BinarySampleSplit.NEGATIVE, _BALANCED_SAMPLES_NEG),
        ("binary_data_imbalanced", BinarySampleSplit.POSITIVE, _IMBALANCED_SAMPLES_POS),
    ],
    indirect=["binary_data_dispatcher"],
)
def test_get_sample(
    mocker: MockerFixture,
    binary_data_dispatcher: BinaryDataTuple,
    split: BinarySampleSplit,
    expected: List[float],
):
    id_to_is_pos, _, id_to_prob_pos = binary_data_dispatcher
    spy_partitioner = mocker.spy(obj=BinarySamplePartitioner, name="partition")
    result = ScoreDistributionSampler.get_sample(
        id_to_is_pos=id_to_is_pos, id_to_prob_pos=id_to_prob_pos, split=split
    )
    assert result == expected
    spy_partitioner.assert_called_once()


@pytest.mark.unit
@pytest.mark.parametrize(
    "binary_data_dispatcher, splits, expected",
    [
        (
            "binary_data_balanced",
            [BinarySampleSplit.ALL],
            {BinarySampleSplit.ALL: _BALANCED_SAMPLES_ALL},
        ),
        (
            "binary_data_balanced",
            [BinarySampleSplit.POSITIVE, BinarySampleSplit.NEGATIVE],
            {
                BinarySampleSplit.POSITIVE: _BALANCED_SAMPLES_POS,
                BinarySampleSplit.NEGATIVE: _BALANCED_SAMPLES_NEG,
            },
        ),
        (
            "binary_data_balanced",
            list(BinarySampleSplit),
            {
                BinarySampleSplit.ALL: _BALANCED_SAMPLES_ALL,
                BinarySampleSplit.POSITIVE: _BALANCED_SAMPLES_POS,
                BinarySampleSplit.NEGATIVE: _BALANCED_SAMPLES_NEG,
            },
        ),
    ],
    indirect=["binary_data_dispatcher"],
)
def test_get_dict_samples(
    mocker: MockerFixture,
    binary_data_dispatcher: BinaryDataTuple,
    splits: List[BinarySampleSplit],
    expected: Mapping[BinarySampleSplit, List[float]],
):
    id_to_is_pos, _, id_to_prob_pos = binary_data_dispatcher
    spy_partitioner = mocker.spy(obj=BinarySamplePartitioner, name="partition")
    result = ScoreDistributionSampler.get_dict_samples(
        id_to_is_pos=id_to_is_pos, id_to_prob_pos=id_to_prob_pos, splits=splits
    )
    assert result == expected
    assert spy_partitioner.call_count == len(splits)
