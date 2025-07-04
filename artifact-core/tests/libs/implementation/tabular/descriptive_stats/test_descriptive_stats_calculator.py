from math import isclose
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest
from artifact_core.libs.implementation.tabular.descriptive_stats.calculator import (
    DescriptiveStatistic,
    DescriptiveStatsCalculator,
)


@pytest.mark.parametrize(
    "df, ls_cts_features, stat, expected_stats, expect_raise_missing, expect_raise_unsupported",
    [
        (
            pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}),
            ["a", "b"],
            DescriptiveStatistic.MEAN,
            {"a": 2.0, "b": 5.0},
            False,
            False,
        ),
        (
            pd.DataFrame({"a": [1, 1, 1], "b": [2, 2, 2]}),
            ["a", "b"],
            DescriptiveStatistic.STD,
            {"a": 0.0, "b": 0.0},
            False,
            False,
        ),
        (
            pd.DataFrame({"a": [1, 1, 1], "b": [2, 2, 2]}),
            ["a", "b"],
            DescriptiveStatistic.VARIANCE,
            {"a": 0.0, "b": 0.0},
            False,
            False,
        ),
        (
            pd.DataFrame({"a": [10, 20, 30], "b": [5, 6, 7]}),
            ["a", "b"],
            DescriptiveStatistic.MEDIAN,
            {"a": 20.0, "b": 6.0},
            False,
            False,
        ),
        (
            pd.DataFrame({"a": [1, 2, 3, 4], "b": [10, 20, 30, 40]}),
            ["a", "b"],
            DescriptiveStatistic.Q1,
            {"a": 1.75, "b": 17.5},
            False,
            False,
        ),
        (
            pd.DataFrame({"a": [1, 2, 3, 4], "b": [10, 20, 30, 40]}),
            ["a", "b"],
            DescriptiveStatistic.Q3,
            {"a": 3.25, "b": 32.5},
            False,
            False,
        ),
        (
            pd.DataFrame({"a": [-1, -2, -3], "b": [100, 200, 300]}),
            ["a", "b"],
            DescriptiveStatistic.MAX,
            {"a": -1.0, "b": 300.0},
            False,
            False,
        ),
        (
            pd.DataFrame({"a": [1, 2, 3]}),
            ["a", "missing_feature"],
            DescriptiveStatistic.MIN,
            {},
            True,
            False,
        ),
        (
            pd.DataFrame({"a": [1, 2, 3]}),
            ["a"],
            "Unsupported",
            {},
            False,
            True,
        ),
    ],
)
def test_descriptive_statistics_compute(
    df: pd.DataFrame,
    ls_cts_features: List[str],
    stat: DescriptiveStatistic,
    expected_stats: Dict[str, float],
    expect_raise_missing: bool,
    expect_raise_unsupported: bool,
):
    if expect_raise_missing:
        with pytest.raises(ValueError, match="Missing columns"):
            DescriptiveStatsCalculator.compute(
                df=df,
                ls_cts_features=ls_cts_features,
                stat=stat,
            )
    elif expect_raise_unsupported:
        with pytest.raises(ValueError, match="Unsupported"):
            DescriptiveStatsCalculator.compute(
                df=df,
                ls_cts_features=ls_cts_features,
                stat=stat,
            )
    else:
        result = DescriptiveStatsCalculator.compute(
            df=df,
            ls_cts_features=ls_cts_features,
            stat=stat,
        )
        for feature, expected in expected_stats.items():
            assert isclose(result[feature], expected, rel_tol=1e-7), (
                f"For feature '{feature}', expected {expected}, got {result[feature]}"
            )


@pytest.mark.parametrize(
    "df_real, df_synthetic, ls_cts_features, stat, "
    + "expected_juxtaposition, expect_raise_missing, expect_raise_unsupported",
    [
        (
            pd.DataFrame({"x": [1, 2, 3]}),
            pd.DataFrame({"x": [4, 5, 6]}),
            ["x"],
            DescriptiveStatistic.MEAN,
            {"x": np.array([2.0, 5.0])},
            False,
            False,
        ),
        (
            pd.DataFrame({"x": [1, 1, 1]}),
            pd.DataFrame({"x": [1, 1, 1]}),
            ["x"],
            DescriptiveStatistic.STD,
            {"x": np.array([0, 0])},
            False,
            False,
        ),
        (
            pd.DataFrame({"x": [1, 1, 1]}),
            pd.DataFrame({"x": [1, 1, 1]}),
            ["x"],
            DescriptiveStatistic.VARIANCE,
            {"x": np.array([0, 0])},
            False,
            False,
        ),
        (
            pd.DataFrame({"y": [1, 2, 3, 4]}),
            pd.DataFrame({"y": [10, 20, 30, 40]}),
            ["y"],
            DescriptiveStatistic.MEDIAN,
            {"y": np.array([2.5, 25])},
            False,
            False,
        ),
        (
            pd.DataFrame({"y": [1, 2, 3, 4]}),
            pd.DataFrame({"y": [10, 20, 30, 40]}),
            ["y"],
            DescriptiveStatistic.Q1,
            {"y": np.array([1.75, 17.5])},
            False,
            False,
        ),
        (
            pd.DataFrame({"y": [1, 2, 3, 4]}),
            pd.DataFrame({"y": [10, 20, 30, 40]}),
            ["y"],
            DescriptiveStatistic.Q3,
            {"y": np.array([3.25, 32.5])},
            False,
            False,
        ),
        (
            pd.DataFrame({"a": [1, 2, 3], "b": [7, 8, 9]}),
            pd.DataFrame({"a": [4, 5, 6], "b": [10, 11, 12]}),
            ["a", "b"],
            DescriptiveStatistic.MIN,
            {"a": np.array([1.0, 4.0]), "b": np.array([7.0, 10.0])},
            False,
            False,
        ),
        (
            pd.DataFrame({"a": [1, 2, 3], "b": [7, 8, 9]}),
            pd.DataFrame({"a": [4, 5, 6], "b": [10, 11, 12]}),
            ["a", "b"],
            DescriptiveStatistic.MAX,
            {"a": np.array([3.0, 6.0]), "b": np.array([9.0, 12.0])},
            False,
            False,
        ),
        (
            pd.DataFrame({"a": [1, 2, 3]}),
            pd.DataFrame({"a": [1, 2, 3]}),
            ["a", "missing_feature"],
            DescriptiveStatistic.MIN,
            {},
            True,
            False,
        ),
        (
            pd.DataFrame({"a": [1, 2, 3]}),
            pd.DataFrame({"a": [1, 2, 3]}),
            ["a"],
            "Unsupported",
            {},
            False,
            True,
        ),
    ],
)
def test_compute_juxtaposition(
    df_real: pd.DataFrame,
    df_synthetic: pd.DataFrame,
    ls_cts_features: List[str],
    stat: DescriptiveStatistic,
    expected_juxtaposition: Dict[str, np.ndarray],
    expect_raise_missing: bool,
    expect_raise_unsupported: bool,
):
    if expect_raise_missing:
        with pytest.raises(ValueError, match="Missing columns"):
            DescriptiveStatsCalculator.compute_juxtaposition(
                df_real=df_real,
                df_synthetic=df_synthetic,
                ls_cts_features=ls_cts_features,
                stat=stat,
            )
    elif expect_raise_unsupported:
        with pytest.raises(ValueError, match="Unsupported"):
            DescriptiveStatsCalculator.compute_juxtaposition(
                df_real=df_real,
                df_synthetic=df_synthetic,
                ls_cts_features=ls_cts_features,
                stat=stat,
            )
    else:
        result = DescriptiveStatsCalculator.compute_juxtaposition(
            df_real=df_real,
            df_synthetic=df_synthetic,
            ls_cts_features=ls_cts_features,
            stat=stat,
        )
        for feature, expected_array in expected_juxtaposition.items():
            np.testing.assert_allclose(
                result[feature],
                expected_array,
                rtol=1e-7,
                err_msg=f"Juxtaposition mismatch for feature '{feature}'",
            )
