from typing import List

import numpy as np
import pandas as pd
import pytest
from artifact_core._libs.artifacts.binary_classification.score_distribution.calculator import (
    ScoreStatsCalculator,
)
from artifact_core._libs.artifacts.binary_classification.score_distribution.partitioner import (
    BinarySampleSplit,
)
from artifact_core._libs.artifacts.binary_classification.score_distribution.sampler import (
    ScoreDistributionSampler,
)
from artifact_core._libs.tools.calculators.descriptive_stats_calculator import (
    DescriptiveStatistic,
    DescriptiveStatsCalculator,
)
from pytest_mock import MockerFixture

from tests._libs.artifacts.binary_classification.conftest import BinaryDataTuple

# Pre-computed expected values for binary_data_balanced fixture:
# - Positives (0,1,4): probs [0.9, 0.4, 0.8]
# - Negatives (2,3,5): probs [0.2, 0.6, 0.1]
# - All: probs [0.9, 0.4, 0.2, 0.6, 0.8, 0.1]
_BALANCED_POS_PROBS = [0.9, 0.4, 0.8]
_BALANCED_NEG_PROBS = [0.2, 0.6, 0.1]
_BALANCED_ALL_PROBS = [0.9, 0.4, 0.2, 0.6, 0.8, 0.1]


@pytest.mark.unit
@pytest.mark.parametrize(
    "binary_data_dispatcher, stats, splits, expected_df",
    [
        (
            "binary_data_balanced",
            [DescriptiveStatistic.MEAN],
            [BinarySampleSplit.ALL],
            pd.DataFrame(
                {"ALL": [np.mean(_BALANCED_ALL_PROBS)]},
                index=["MEAN"],
            ),
        ),
        (
            "binary_data_balanced",
            [DescriptiveStatistic.MEAN, DescriptiveStatistic.STD],
            [BinarySampleSplit.POSITIVE, BinarySampleSplit.NEGATIVE],
            pd.DataFrame(
                {
                    "POSITIVE": [np.mean(_BALANCED_POS_PROBS), np.std(_BALANCED_POS_PROBS, ddof=1)],
                    "NEGATIVE": [np.mean(_BALANCED_NEG_PROBS), np.std(_BALANCED_NEG_PROBS, ddof=1)],
                },
                index=["MEAN", "STD"],
            ),
        ),
        (
            "binary_data_balanced",
            [DescriptiveStatistic.MIN, DescriptiveStatistic.MAX],
            list(BinarySampleSplit),
            pd.DataFrame(
                {
                    "ALL": [min(_BALANCED_ALL_PROBS), max(_BALANCED_ALL_PROBS)],
                    "POSITIVE": [min(_BALANCED_POS_PROBS), max(_BALANCED_POS_PROBS)],
                    "NEGATIVE": [min(_BALANCED_NEG_PROBS), max(_BALANCED_NEG_PROBS)],
                },
                index=["MIN", "MAX"],
            ),
        ),
    ],
    indirect=["binary_data_dispatcher"],
)
def test_compute_as_df(
    mocker: MockerFixture,
    binary_data_dispatcher: BinaryDataTuple,
    stats: List[DescriptiveStatistic],
    splits: List[BinarySampleSplit],
    expected_df: pd.DataFrame,
):
    id_to_is_pos, _, id_to_prob_pos = binary_data_dispatcher
    spy_sampler = mocker.spy(obj=ScoreDistributionSampler, name="get_sample")
    spy_stats = mocker.spy(obj=DescriptiveStatsCalculator, name="compute_sr_stats")
    result = ScoreStatsCalculator.compute_as_df(
        id_to_is_pos=id_to_is_pos, id_to_prob_pos=id_to_prob_pos, stats=stats, splits=splits
    )
    pd.testing.assert_frame_equal(result, expected_df)
    assert spy_sampler.call_count == len(splits)
    assert spy_stats.call_count == len(splits)


@pytest.mark.unit
@pytest.mark.parametrize(
    "binary_data_dispatcher", ["binary_data_balanced", "binary_data_imbalanced"], indirect=True
)
def test_compute_as_dict(binary_data_dispatcher: BinaryDataTuple):
    id_to_is_pos, _, id_to_prob_pos = binary_data_dispatcher
    stats = [DescriptiveStatistic.MEAN, DescriptiveStatistic.STD]
    splits = [BinarySampleSplit.POSITIVE, BinarySampleSplit.NEGATIVE]
    result = ScoreStatsCalculator.compute_as_dict(
        id_to_is_pos=id_to_is_pos, id_to_prob_pos=id_to_prob_pos, stats=stats, splits=splits
    )
    assert set(result.keys()) == set(stats)
    for stat, split_dict in result.items():
        assert set(split_dict.keys()) == set(splits)
        for split, value in split_dict.items():
            assert isinstance(value, float)


@pytest.mark.unit
@pytest.mark.parametrize(
    "binary_data_dispatcher, split",
    [
        ("binary_data_balanced", BinarySampleSplit.ALL),
        ("binary_data_balanced", BinarySampleSplit.POSITIVE),
        ("binary_data_balanced", BinarySampleSplit.NEGATIVE),
    ],
    indirect=["binary_data_dispatcher"],
)
def test_compute_by_split(binary_data_dispatcher: BinaryDataTuple, split: BinarySampleSplit):
    id_to_is_pos, _, id_to_prob_pos = binary_data_dispatcher
    stats = [DescriptiveStatistic.MEAN, DescriptiveStatistic.MEDIAN]
    result = ScoreStatsCalculator.compute_by_split(
        id_to_is_pos=id_to_is_pos, id_to_prob_pos=id_to_prob_pos, stats=stats, split=split
    )
    assert set(result.keys()) == set(stats)
    for stat, value in result.items():
        assert isinstance(value, float)


@pytest.mark.unit
@pytest.mark.parametrize(
    "binary_data_dispatcher, stat",
    [
        ("binary_data_balanced", DescriptiveStatistic.MEAN),
        ("binary_data_balanced", DescriptiveStatistic.STD),
        ("binary_data_balanced", DescriptiveStatistic.MEDIAN),
    ],
    indirect=["binary_data_dispatcher"],
)
def test_compute_by_stat(binary_data_dispatcher: BinaryDataTuple, stat: DescriptiveStatistic):
    id_to_is_pos, _, id_to_prob_pos = binary_data_dispatcher
    splits = [BinarySampleSplit.POSITIVE, BinarySampleSplit.NEGATIVE]
    result = ScoreStatsCalculator.compute_by_stat(
        id_to_is_pos=id_to_is_pos, id_to_prob_pos=id_to_prob_pos, stat=stat, splits=splits
    )
    assert set(result.keys()) == set(splits)
    for split, value in result.items():
        assert isinstance(value, float)


@pytest.mark.unit
def test_compute_as_dict_empty_stats():
    result = ScoreStatsCalculator.compute_as_dict(
        id_to_is_pos={0: True}, id_to_prob_pos={0: 0.5}, stats=[], splits=[BinarySampleSplit.ALL]
    )
    assert result == {}


@pytest.mark.unit
def test_compute_as_dict_empty_splits():
    result = ScoreStatsCalculator.compute_as_dict(
        id_to_is_pos={0: True},
        id_to_prob_pos={0: 0.5},
        stats=[DescriptiveStatistic.MEAN],
        splits=[],
    )
    assert result == {}
