from typing import List, cast

import pandas as pd
import pytest
from artifact_core._libs.tools.calculators.descriptive_stats_calculator import (
    DescriptiveStatistic,
    DescriptiveStatsCalculator,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    "data, stat, expected",
    [
        ([1.0, 2.0, 3.0, 4.0, 5.0], DescriptiveStatistic.MEAN, 3.0),
        ([1.0, 2.0, 3.0, 4.0, 5.0], DescriptiveStatistic.MEDIAN, 3.0),
        ([1.0, 2.0, 3.0, 4.0, 5.0], DescriptiveStatistic.MIN, 1.0),
        ([1.0, 2.0, 3.0, 4.0, 5.0], DescriptiveStatistic.MAX, 5.0),
        ([1.0, 1.0, 1.0], DescriptiveStatistic.STD, 0.0),
        ([1.0, 1.0, 1.0], DescriptiveStatistic.VARIANCE, 0.0),
        ([1.0, 2.0, 3.0, 4.0], DescriptiveStatistic.Q1, 1.75),
        ([1.0, 2.0, 3.0, 4.0], DescriptiveStatistic.Q3, 3.25),
    ],
)
def test_compute_stat(data: List[float], stat: DescriptiveStatistic, expected: float):
    sr = pd.Series(data)
    result = DescriptiveStatsCalculator.compute_stat(sr_cts_data=sr, stat=stat)
    assert result == pytest.approx(expected=expected)


@pytest.mark.unit
def test_compute_stat_unsupported_raises():
    from enum import Enum

    class FakeStat(Enum):
        UNSUPPORTED = "unsupported"

    sr = pd.Series([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="Unsupported statistic"):
        DescriptiveStatsCalculator.compute_stat(
            sr_cts_data=sr, stat=cast(DescriptiveStatistic, FakeStat.UNSUPPORTED)
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "data, stats",
    [
        ([1.0, 2.0, 3.0], [DescriptiveStatistic.MEAN, DescriptiveStatistic.STD]),
        ([1.0, 2.0, 3.0, 4.0, 5.0], list(DescriptiveStatistic)),
    ],
)
def test_compute_dict_stats(data: List[float], stats: List[DescriptiveStatistic]):
    sr = pd.Series(data)
    result = DescriptiveStatsCalculator.compute_dict_stats(sr_cts_data=sr, stats=stats)
    assert set(result.keys()) == set(stats)
    for stat in stats:
        assert isinstance(result[stat], float)


@pytest.mark.unit
@pytest.mark.parametrize(
    "data, stats",
    [
        ([1.0, 2.0, 3.0], [DescriptiveStatistic.MEAN, DescriptiveStatistic.STD]),
        ([1.0, 2.0, 3.0, 4.0, 5.0], [DescriptiveStatistic.MIN, DescriptiveStatistic.MAX]),
    ],
)
def test_compute_sr_stats(data: List[float], stats: List[DescriptiveStatistic]):
    sr = pd.Series(data)
    result = DescriptiveStatsCalculator.compute_sr_stats(sr_cts_data=sr, stats=stats)
    assert isinstance(result, pd.Series)
    assert len(result) == len(stats)
    for stat in stats:
        assert stat.name in result.index


@pytest.mark.unit
def test_compute_stat_with_nan():
    sr = pd.Series([1.0, 2.0, float("nan"), 4.0, 5.0])
    result = DescriptiveStatsCalculator.compute_stat(sr_cts_data=sr, stat=DescriptiveStatistic.MEAN)
    assert result == pytest.approx(expected=3.0)
