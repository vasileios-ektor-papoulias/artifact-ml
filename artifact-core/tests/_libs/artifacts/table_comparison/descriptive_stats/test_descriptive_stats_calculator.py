from math import isclose
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest
from artifact_core._base.typing.artifact_result import Array
from artifact_core._libs.artifacts.table_comparison.descriptive_stats.calculator import (
    DescriptiveStatistic,
    TableStatsCalculator,
)
from artifact_core._libs.tools.calculators.descriptive_stats_calculator import (
    DescriptiveStatsCalculator,
)
from artifact_core._libs.tools.calculators.score_juxtaposition_calculator import (
    ScoreJuxtapositionCalculator,
)
from pytest_mock import MockerFixture


@pytest.mark.unit
@pytest.mark.parametrize(
    "df, ls_cts_features, stat, expected_stats",
    [
        (
            pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}),
            ["a", "b"],
            DescriptiveStatistic.MEAN,
            {"a": 2.0, "b": 5.0},
        ),
        (
            pd.DataFrame({"a": [1, 1, 1], "b": [2, 2, 2]}),
            ["a", "b"],
            DescriptiveStatistic.STD,
            {"a": 0.0, "b": 0.0},
        ),
        (
            pd.DataFrame({"a": [1, 1, 1], "b": [2, 2, 2]}),
            ["a", "b"],
            DescriptiveStatistic.VARIANCE,
            {"a": 0.0, "b": 0.0},
        ),
        (
            pd.DataFrame({"a": [10, 20, 30], "b": [5, 6, 7]}),
            ["a", "b"],
            DescriptiveStatistic.MEDIAN,
            {"a": 20.0, "b": 6.0},
        ),
        (
            pd.DataFrame({"a": [1, 2, 3, 4], "b": [10, 20, 30, 40]}),
            ["a", "b"],
            DescriptiveStatistic.Q1,
            {"a": 1.75, "b": 17.5},
        ),
        (
            pd.DataFrame({"a": [1, 2, 3, 4], "b": [10, 20, 30, 40]}),
            ["a", "b"],
            DescriptiveStatistic.Q3,
            {"a": 3.25, "b": 32.5},
        ),
        (
            pd.DataFrame({"a": [-1, -2, -3], "b": [100, 200, 300]}),
            ["a", "b"],
            DescriptiveStatistic.MAX,
            {"a": -1.0, "b": 300.0},
        ),
    ],
)
def test_compute(
    mocker: MockerFixture,
    df: pd.DataFrame,
    ls_cts_features: List[str],
    stat: DescriptiveStatistic,
    expected_stats: Dict[str, float],
):
    spy = mocker.spy(obj=DescriptiveStatsCalculator, name="compute_stat")
    result = TableStatsCalculator.compute(df=df, cts_features=ls_cts_features, stat=stat)
    assert spy.call_count == len(ls_cts_features)
    for call in spy.call_args_list:
        assert call.kwargs["stat"] == stat
    for feature, expected in expected_stats.items():
        assert isclose(result[feature], expected, rel_tol=1e-7)


@pytest.mark.unit
def test_compute_raises_on_missing_columns(mocker: MockerFixture):
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ValueError, match="Missing columns"):
        TableStatsCalculator.compute(
            df=df, cts_features=["a", "missing_feature"], stat=DescriptiveStatistic.MIN
        )


@pytest.mark.unit
def test_compute_raises_on_unsupported_stat(mocker: MockerFixture):
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ValueError, match="Unsupported"):
        TableStatsCalculator.compute(
            df=df,
            cts_features=["a"],
            stat="Unsupported",  # type: ignore
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "df_real, df_synthetic, ls_cts_features, stat, expected_juxtaposition",
    [
        (
            pd.DataFrame({"x": [1, 2, 3]}),
            pd.DataFrame({"x": [4, 5, 6]}),
            ["x"],
            DescriptiveStatistic.MEAN,
            {"x": np.array([2.0, 5.0])},
        ),
        (
            pd.DataFrame({"x": [1, 1, 1]}),
            pd.DataFrame({"x": [1, 1, 1]}),
            ["x"],
            DescriptiveStatistic.STD,
            {"x": np.array([0, 0])},
        ),
        (
            pd.DataFrame({"x": [1, 1, 1]}),
            pd.DataFrame({"x": [1, 1, 1]}),
            ["x"],
            DescriptiveStatistic.VARIANCE,
            {"x": np.array([0, 0])},
        ),
        (
            pd.DataFrame({"y": [1, 2, 3, 4]}),
            pd.DataFrame({"y": [10, 20, 30, 40]}),
            ["y"],
            DescriptiveStatistic.MEDIAN,
            {"y": np.array([2.5, 25])},
        ),
        (
            pd.DataFrame({"y": [1, 2, 3, 4]}),
            pd.DataFrame({"y": [10, 20, 30, 40]}),
            ["y"],
            DescriptiveStatistic.Q1,
            {"y": np.array([1.75, 17.5])},
        ),
        (
            pd.DataFrame({"y": [1, 2, 3, 4]}),
            pd.DataFrame({"y": [10, 20, 30, 40]}),
            ["y"],
            DescriptiveStatistic.Q3,
            {"y": np.array([3.25, 32.5])},
        ),
        (
            pd.DataFrame({"a": [1, 2, 3], "b": [7, 8, 9]}),
            pd.DataFrame({"a": [4, 5, 6], "b": [10, 11, 12]}),
            ["a", "b"],
            DescriptiveStatistic.MIN,
            {"a": np.array([1.0, 4.0]), "b": np.array([7.0, 10.0])},
        ),
        (
            pd.DataFrame({"a": [1, 2, 3], "b": [7, 8, 9]}),
            pd.DataFrame({"a": [4, 5, 6], "b": [10, 11, 12]}),
            ["a", "b"],
            DescriptiveStatistic.MAX,
            {"a": np.array([3.0, 6.0]), "b": np.array([9.0, 12.0])},
        ),
    ],
)
def test_compute_juxtaposition(
    mocker: MockerFixture,
    df_real: pd.DataFrame,
    df_synthetic: pd.DataFrame,
    ls_cts_features: List[str],
    stat: DescriptiveStatistic,
    expected_juxtaposition: Dict[str, Array],
):
    spy_juxtaposition = mocker.spy(
        obj=ScoreJuxtapositionCalculator, name="juxtapose_score_collections"
    )

    result = TableStatsCalculator.compute_juxtaposition(
        df_real=df_real,
        df_synthetic=df_synthetic,
        cts_features=ls_cts_features,
        stat=stat,
    )
    spy_juxtaposition.assert_called_once()
    assert list(spy_juxtaposition.call_args.kwargs["keys"]) == ls_cts_features
    for feature, expected_array in expected_juxtaposition.items():
        np.testing.assert_allclose(
            result[feature],
            expected_array,
            rtol=1e-7,
            err_msg=f"Juxtaposition mismatch for feature '{feature}'",
        )


@pytest.mark.unit
def test_compute_juxtaposition_raises_on_missing_columns(mocker: MockerFixture):
    df_real = pd.DataFrame({"a": [1, 2, 3]})
    df_synthetic = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ValueError, match="Missing columns"):
        TableStatsCalculator.compute_juxtaposition(
            df_real=df_real,
            df_synthetic=df_synthetic,
            cts_features=["a", "missing_feature"],
            stat=DescriptiveStatistic.MIN,
        )


@pytest.mark.unit
def test_compute_juxtaposition_raises_on_unsupported_stat(mocker: MockerFixture):
    df_real = pd.DataFrame({"a": [1, 2, 3]})
    df_synthetic = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ValueError, match="Unsupported"):
        TableStatsCalculator.compute_juxtaposition(
            df_real=df_real,
            df_synthetic=df_synthetic,
            cts_features=["a"],
            stat="Unsupported",  # type: ignore
        )
