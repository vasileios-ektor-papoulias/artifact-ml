import math
from math import isclose
from typing import Dict, List

import pandas as pd
import pytest
from artifact_core.libs.implementation.js.js import JSDistanceCalculator


@pytest.mark.parametrize(
    "df_real, df_synth, ls_cts_features, ls_cat_features, cat_unique_map, n_bins, "
    + "categorical_only, expected, expect_raise_missing, expect_raise_overlap",
    [
        (
            pd.DataFrame({"color": ["red", "blue", "red", "green"]}),
            pd.DataFrame({"color": ["red", "blue", "red", "green"]}),
            [],
            ["color"],
            {"color": ["red", "blue", "green"]},
            5,
            True,
            0.0,
            False,
            False,
        ),
        (
            pd.DataFrame({"color": ["red", "blue", "red", "red"]}),
            pd.DataFrame({"color": ["blue", "blue", "green", "green"]}),
            [],
            ["color"],
            {"color": ["red", "blue", "green"]},
            5,
            True,
            None,
            False,
            False,
        ),
        (
            pd.DataFrame({"color": ["red", "blue"]}),
            pd.DataFrame({"color": ["green", "orange"]}),
            [],
            ["color"],
            {"color": ["red", "blue", "green", "orange"]},
            5,
            True,
            math.sqrt(math.log(2)),
            False,
            False,
        ),
        (
            pd.DataFrame({"height": [1.0, 2.0, 3.0, 4.0]}),
            pd.DataFrame({"height": [1.0, 2.0, 3.0, 4.0]}),
            ["height"],
            [],
            {},
            8,
            False,
            0.0,
            False,
            False,
        ),
        (
            pd.DataFrame({"height": [1.0, 2.0, 3.0, 4.0]}),
            pd.DataFrame({"height": [5.0, 6.0, 7.0, 8.0]}),
            ["height"],
            [],
            {},
            8,
            False,
            math.sqrt(math.log(2)),
            False,
            False,
        ),
        (
            pd.DataFrame({"height": [1.0, 2.0, 3.0, 4.0]}),
            pd.DataFrame({"height": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]}),
            ["height"],
            [],
            {},
            4,
            False,
            None,
            False,
            False,
        ),
        (
            pd.DataFrame({"color": ["red", "green", "blue", "blue"], "age": [10, 20, 20, 40]}),
            pd.DataFrame({"color": ["red", "green", "blue", "blue"], "age": [10, 20, 20, 40]}),
            ["age"],
            ["color"],
            {"color": ["red", "green", "blue"]},
            2,
            False,
            0.0,
            False,
            False,
        ),
        (
            pd.DataFrame({"color": ["red", "green", "blue", "blue"], "age": [10, 20, 20, 40]}),
            pd.DataFrame({"color": ["green", "green", "blue", "red"], "age": [100, 200, 200, 400]}),
            ["age"],
            ["color"],
            {"color": ["red", "green", "blue"]},
            2,
            False,
            None,
            False,
            False,
        ),
        (
            pd.DataFrame({"color": ["red", "green", "blue", "blue"], "age": [10, 20, 20, 40]}),
            pd.DataFrame({"color": ["green", "green", "blue", "red"], "age": [100, 200, 200, 400]}),
            ["age", "height"],
            ["color"],
            {"color": ["red", "green", "blue"]},
            2,
            False,
            None,
            True,
            False,
        ),
        (
            pd.DataFrame({"color": ["red", "green", "blue", "blue"], "age": [10, 20, 20, 40]}),
            pd.DataFrame({"color": ["green", "green", "blue", "red"], "age": [100, 200, 200, 400]}),
            ["age", "height"],
            ["color", "type"],
            {"color": ["red", "green", "blue"]},
            2,
            False,
            None,
            True,
            False,
        ),
        (
            pd.DataFrame({"color": ["red", "green", "blue", "blue"], "age": [10, 20, 20, 40]}),
            pd.DataFrame({"color": ["green", "green", "blue", "red"], "age": [100, 200, 200, 400]}),
            ["age"],
            ["color", "age"],
            {"color": ["red", "green", "blue"]},
            2,
            False,
            None,
            False,
            True,
        ),
        (
            pd.DataFrame({"color": ["red", "green", "blue", "blue"], "age": [10, 20, 20, 40]}),
            pd.DataFrame({"color": ["green", "green", "blue", "red"], "age": [100, 200, 200, 400]}),
            ["age", "color"],
            ["color"],
            {"color": ["red", "green", "blue"]},
            2,
            False,
            None,
            False,
            True,
        ),
        (
            pd.DataFrame({"color": ["red", "green", "blue", "blue"], "age": [10, 20, 20, 40]}),
            pd.DataFrame({"color": ["green", "green", "blue", "red"], "age": [100, 200, 200, 400]}),
            ["age", "color"],
            ["color"],
            {"color": ["red", "green", "blue"]},
            2,
            True,
            None,
            False,
            False,
        ),
    ],
)
def test_compute_mean_js(
    df_real: pd.DataFrame,
    df_synth: pd.DataFrame,
    ls_cts_features: List[str],
    ls_cat_features: List[str],
    cat_unique_map: Dict[str, List[str]],
    n_bins: int,
    categorical_only: bool,
    expected: float,
    expect_raise_missing: bool,
    expect_raise_overlap: bool,
):
    if expect_raise_missing:
        with pytest.raises(ValueError, match="Missing columns"):
            JSDistanceCalculator.compute_mean_js(
                df_real=df_real,
                df_synthetic=df_synth,
                ls_cts_features=ls_cts_features,
                ls_cat_features=ls_cat_features,
                cat_unique_map=cat_unique_map,
                n_bins_cts_histogram=n_bins,
                categorical_only=categorical_only,
            )
    elif expect_raise_overlap:
        with pytest.raises(ValueError, match="Continuous and categorical features overlap"):
            JSDistanceCalculator.compute_mean_js(
                df_real=df_real,
                df_synthetic=df_synth,
                ls_cts_features=ls_cts_features,
                ls_cat_features=ls_cat_features,
                cat_unique_map=cat_unique_map,
                n_bins_cts_histogram=n_bins,
                categorical_only=categorical_only,
            )
    else:
        mean_js_distance = JSDistanceCalculator.compute_mean_js(
            df_real=df_real,
            df_synthetic=df_synth,
            ls_cts_features=ls_cts_features,
            ls_cat_features=ls_cat_features,
            cat_unique_map=cat_unique_map,
            n_bins_cts_histogram=n_bins,
            categorical_only=categorical_only,
        )

        if expected is not None:
            assert isclose(mean_js_distance, expected, rel_tol=1e-7), (
                f"Expected JS distance {expected}, got {mean_js_distance}"
            )
        else:
            assert 0.0 < mean_js_distance <= math.sqrt(math.log(2)), (
                f"Expected 0 < JS distance <= sqrt(log(2)), got {mean_js_distance}"
            )


@pytest.mark.parametrize(
    "df_real, df_synth, ls_cts_features, ls_cat_features, cat_unique_map, "
    "n_bins, categorical_only, expected_js_dict, expect_raise_missing, expect_raise_overlap",
    [
        (
            pd.DataFrame({"color": ["red", "blue", "red", "green"]}),
            pd.DataFrame({"color": ["red", "blue", "red", "green"]}),
            [],
            ["color"],
            {"color": ["red", "blue", "green"]},
            5,
            True,
            {"color": 0.0},
            False,
            False,
        ),
        (
            pd.DataFrame({"color": ["red", "blue"]}),
            pd.DataFrame({"color": ["green", "orange"]}),
            [],
            ["color"],
            {"color": ["red", "blue", "green", "orange"]},
            5,
            True,
            {"color": math.sqrt(math.log(2))},
            False,
            False,
        ),
        (
            pd.DataFrame({"height": [1.0, 2.0, 3.0, 4.0]}),
            pd.DataFrame({"height": [1.0, 2.0, 3.0, 4.0]}),
            ["height"],
            [],
            {},
            8,
            False,
            {"height": 0.0},
            False,
            False,
        ),
        (
            pd.DataFrame({"height": [1.0, 2.0, 3.0, 4.0]}),
            pd.DataFrame({"height": [5.0, 6.0, 7.0, 8.0]}),
            ["height"],
            [],
            {},
            8,
            False,
            {"height": math.sqrt(math.log(2))},
            False,
            False,
        ),
        (
            pd.DataFrame({"height": [1.0, 2.0, 3.0, 4.0]}),
            pd.DataFrame({"height": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]}),
            ["height"],
            [],
            {},
            4,
            False,
            {"height": None},
            False,
            False,
        ),
        (
            pd.DataFrame({"color": ["red", "green"], "age": [10, 20]}),
            pd.DataFrame({"color": ["red", "green"], "age": [10, 20]}),
            ["age"],
            ["color"],
            {"color": ["red", "green"]},
            2,
            False,
            {"color": 0.0, "age": 0.0},
            False,
            False,
        ),
        (
            pd.DataFrame({"color": ["red", "green"], "age": [10, 20]}),
            pd.DataFrame({"color": ["blue", "yellow"], "age": [30, 40]}),
            ["age"],
            ["color"],
            {"color": ["red", "green", "blue", "yellow"]},
            2,
            False,
            {"color": None, "age": None},
            False,
            False,
        ),
        (
            pd.DataFrame({"age": [10, 20]}),
            pd.DataFrame({"age": [30, 40]}),
            ["age", "height"],
            [],
            {},
            2,
            False,
            {},
            True,
            False,
        ),
        (
            pd.DataFrame({"color": ["red", "green"], "age": [10, 20]}),
            pd.DataFrame({"color": ["blue", "yellow"], "age": [30, 40]}),
            ["age", "color"],
            ["color"],
            {"color": ["red", "green", "blue", "yellow"]},
            2,
            False,
            {},
            False,
            True,
        ),
    ],
)
def test_compute_dict_js(
    df_real: pd.DataFrame,
    df_synth: pd.DataFrame,
    ls_cts_features: List[str],
    ls_cat_features: List[str],
    cat_unique_map: Dict[str, List[str]],
    n_bins: int,
    categorical_only: bool,
    expected_js_dict: Dict[str, float],
    expect_raise_missing: bool,
    expect_raise_overlap: bool,
):
    if expect_raise_missing:
        with pytest.raises(ValueError, match="Missing columns"):
            JSDistanceCalculator.compute_dict_js(
                df_real=df_real,
                df_synthetic=df_synth,
                ls_cts_features=ls_cts_features,
                ls_cat_features=ls_cat_features,
                cat_unique_map=cat_unique_map,
                n_bins_cts_histogram=n_bins,
                categorical_only=categorical_only,
            )
    elif expect_raise_overlap:
        with pytest.raises(ValueError, match="Continuous and categorical features overlap"):
            JSDistanceCalculator.compute_dict_js(
                df_real=df_real,
                df_synthetic=df_synth,
                ls_cts_features=ls_cts_features,
                ls_cat_features=ls_cat_features,
                cat_unique_map=cat_unique_map,
                n_bins_cts_histogram=n_bins,
                categorical_only=categorical_only,
            )
    else:
        dict_js = JSDistanceCalculator.compute_dict_js(
            df_real=df_real,
            df_synthetic=df_synth,
            ls_cts_features=ls_cts_features,
            ls_cat_features=ls_cat_features,
            cat_unique_map=cat_unique_map,
            n_bins_cts_histogram=n_bins,
            categorical_only=categorical_only,
        )

        assert dict_js.keys() == expected_js_dict.keys(), (
            f"Expected JS keys {expected_js_dict.keys()}, got {dict_js.keys()}"
        )

        for feature, expected_js in expected_js_dict.items():
            js_distance = dict_js[feature]

            if expected_js is not None:
                assert isclose(js_distance, expected_js, rel_tol=1e-7), (
                    f"For feature '{feature}', expected JS distance {expected_js}, "
                    f"got {js_distance}"
                )
            else:
                assert 0.0 < js_distance <= math.sqrt(math.log(2)), (
                    f"For feature '{feature}', expected JS distance > 0 and <= sqrt(log(2)), "
                    f"got {js_distance}"
                )
