from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pytest
from artifact_core._libs.resources_spec.tabular.protocol import (
    TabularDataDType,
)
from artifact_core._libs.resources_spec.tabular.spec import TabularDataSpec


@pytest.mark.unit
@pytest.mark.parametrize(
    "ls_cts_features, ls_cat_features, "
    + "expected_exception, "
    + "expected_ls_features, expected_n_features, "
    + "expected_ls_cts_features, expected_n_cts_features, expected_dict_cts_dtypes, "
    + "expected_ls_cat_features, expected_n_cat_features, expected_dict_cat_dtypes, "
    + "expected_cat_unique_map, expected_cat_unique_count_map, expected_ls_n_cat",
    [
        (
            ["cts_1", "cts_2"],
            ["cat_1", "cat_2"],
            None,
            ["cts_1", "cts_2", "cat_1", "cat_2"],
            4,
            ["cts_1", "cts_2"],
            2,
            {"cts_1": float, "cts_2": float},
            ["cat_1", "cat_2"],
            2,
            {"cat_1": str, "cat_2": str},
            {"cat_1": [], "cat_2": []},
            {"cat_1": 0, "cat_2": 0},
            [0, 0],
        ),
        (
            ["cts_1", "cts_2"],
            [],
            None,
            ["cts_1", "cts_2"],
            2,
            ["cts_1", "cts_2"],
            2,
            {"cts_1": float, "cts_2": float},
            [],
            0,
            {},
            {},
            {},
            [],
        ),
        (
            [],
            ["cat_1", "cat_2"],
            None,
            ["cat_1", "cat_2"],
            2,
            [],
            0,
            {},
            ["cat_1", "cat_2"],
            2,
            {"cat_1": str, "cat_2": str},
            {"cat_1": [], "cat_2": []},
            {"cat_1": 0, "cat_2": 0},
            [0, 0],
        ),
        (
            [],
            [],
            None,
            [],
            0,
            [],
            0,
            {},
            [],
            0,
            {},
            {},
            {},
            [],
        ),
        (
            None,
            None,
            None,
            [],
            0,
            [],
            0,
            {},
            [],
            0,
            {},
            {},
            {},
            [],
        ),
        (
            ["cts_1", "cts_2"],
            ["cts_1"],
            ValueError("Categorical and continuous features overlap: {'cts_1'}"),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ),
        (
            ["cts_1", "cts_2"],
            ["cts_1", "cat_1"],
            ValueError("Categorical and continuous features overlap: {'cts_1'}"),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ),
        (
            ["cts_1", "cts_2", "cts_3", "cat_1"],
            ["cat_1", "cat_2", "cat_3"],
            ValueError("Categorical and continuous features overlap: {'cat_1'}"),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ),
    ],
)
def test_build(
    ls_cts_features: Optional[List[str]],
    ls_cat_features: Optional[List[str]],
    expected_exception: Optional[BaseException],
    expected_ls_features: Optional[List[str]],
    expected_n_features: Optional[int],
    expected_ls_cts_features: Optional[List[str]],
    expected_n_cts_features: Optional[int],
    expected_dict_cts_dtypes: Optional[Dict[str, TabularDataDType]],
    expected_ls_cat_features: Optional[List[str]],
    expected_n_cat_features: Optional[int],
    expected_dict_cat_dtypes: Optional[Dict[str, TabularDataDType]],
    expected_cat_unique_map: Optional[Dict[str, List[str]]],
    expected_cat_unique_count_map: Optional[Dict[str, int]],
    expected_ls_n_cat: Optional[List[int]],
):
    if expected_exception is not None:
        expected_exception_type = type(expected_exception)
        expected_exception_message = expected_exception.args[0]
        with pytest.raises(
            expected_exception_type,
            match=expected_exception_message,
        ):
            _ = TabularDataSpec.build(
                ls_cts_features=ls_cts_features,
                ls_cat_features=ls_cat_features,
            )
    else:
        spec = TabularDataSpec.build(
            ls_cts_features=ls_cts_features,
            ls_cat_features=ls_cat_features,
        )
        assert spec.ls_features == expected_ls_features, (
            f"Expected ls_features: {expected_ls_features} got {spec.ls_features}"
        )
        assert spec.n_features == expected_n_features, (
            f"Expected n_features: {expected_n_features} got {spec.n_features}"
        )
        assert spec.ls_cts_features == expected_ls_cts_features, (
            f"Expected ls_cts_features: {expected_ls_cts_features} got {spec.ls_cts_features}"
        )
        assert spec.n_cts_features == expected_n_cts_features, (
            f"Expected n_cts_features: {expected_n_cts_features} got {spec.n_cts_features}"
        )
        assert spec.dict_cts_dtypes == expected_dict_cts_dtypes, (
            f"Expected dict_cts_dtypes: {expected_dict_cts_dtypes} got {spec.dict_cts_dtypes}"
        )
        assert spec.ls_cat_features == expected_ls_cat_features, (
            f"Expected ls_cat_features: {expected_ls_cat_features} got {spec.ls_cat_features}"
        )
        assert spec.n_cat_features == expected_n_cat_features, (
            f"Expected n_cat_features: {expected_n_cat_features} got {spec.n_cat_features}"
        )
        assert spec.dict_cat_dtypes == expected_dict_cat_dtypes, (
            f"Expected dict_cat_dtypes: {expected_dict_cat_dtypes} got {spec.dict_cat_dtypes}"
        )
        assert spec.cat_unique_map == expected_cat_unique_map, (
            f"Expected cat_unique_map: {expected_cat_unique_map} got {spec.cat_unique_map}"
        )
        assert spec.cat_unique_count_map == expected_cat_unique_count_map, (
            f"Expected cat_unique_count_map: {expected_cat_unique_count_map}"
            + f"got {spec.cat_unique_count_map}"
        )
        assert spec.ls_n_cat == expected_ls_n_cat, (
            f"Expected ls_n_cat: {expected_ls_n_cat} got {spec.ls_n_cat}"
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "df_dispatcher, ls_cts_features, ls_cat_features, "
    + "expected_exception, "
    + "expected_ls_features, expected_n_features, "
    + "expected_ls_cts_features, expected_n_cts_features, expected_dict_cts_dtypes, "
    + "expected_ls_cat_features, expected_n_cat_features, expected_dict_cat_dtypes, "
    + "expected_cat_unique_map, expected_cat_unique_count_map, expected_ls_n_cat",
    [
        (
            "simple_df",
            None,
            None,
            None,
            ["num1", "num2", "cat1", "cat2"],
            4,
            ["num1", "num2"],
            2,
            {"num1": np.float64, "num2": np.float64},
            ["cat1", "cat2"],
            2,
            {"cat1": np.object_, "cat2": np.object_},
            {"cat1": ["A", "B", "C"], "cat2": ["X", "Y", "Z"]},
            {"cat1": 3, "cat2": 3},
            [3, 3],
        ),
        (
            "simple_df",
            ["num1"],
            None,
            None,
            ["num1", "num2", "cat1", "cat2"],
            4,
            ["num1", "num2"],
            2,
            {"num1": np.float64, "num2": np.float64},
            ["cat1", "cat2"],
            2,
            {"cat1": np.object_, "cat2": np.object_},
            {"cat1": ["A", "B", "C"], "cat2": ["X", "Y", "Z"]},
            {"cat1": 3, "cat2": 3},
            [3, 3],
        ),
        (
            "simple_df",
            None,
            ["cat1"],
            None,
            ["num1", "num2", "cat1", "cat2"],
            4,
            ["num1", "num2"],
            2,
            {"num1": np.float64, "num2": np.float64},
            ["cat1", "cat2"],
            2,
            {"cat1": np.object_, "cat2": np.object_},
            {"cat1": ["A", "B", "C"], "cat2": ["X", "Y", "Z"]},
            {"cat1": 3, "cat2": 3},
            [3, 3],
        ),
        (
            "simple_df",
            ["num1"],
            ["cat1"],
            None,
            ["num1", "num2", "cat1", "cat2"],
            4,
            ["num1", "num2"],
            2,
            {"num1": np.float64, "num2": np.float64},
            ["cat1", "cat2"],
            2,
            {"cat1": np.object_, "cat2": np.object_},
            {"cat1": ["A", "B", "C"], "cat2": ["X", "Y", "Z"]},
            {"cat1": 3, "cat2": 3},
            [3, 3],
        ),
        (
            "simple_df",
            None,
            ["num1"],
            None,
            ["num1", "num2", "cat1", "cat2"],
            4,
            ["num2"],
            1,
            {"num2": np.float64},
            ["num1", "cat1", "cat2"],
            3,
            {"num1": np.float64, "cat1": np.object_, "cat2": np.object_},
            {
                "num1": ["1.0", "2.0", "3.0", "4.0", "5.0"],
                "cat1": ["A", "B", "C"],
                "cat2": ["X", "Y", "Z"],
            },
            {"num1": 5, "cat1": 3, "cat2": 3},
            [5, 3, 3],
        ),
        (
            "simple_df",
            ["cat1"],
            None,
            None,
            ["num1", "num2", "cat1", "cat2"],
            4,
            ["num1", "num2", "cat1"],
            3,
            {
                "num1": np.float64,
                "num2": np.float64,
                "cat1": np.object_,
            },
            ["cat2"],
            1,
            {"cat2": np.object_},
            {"cat2": ["X", "Y", "Z"]},
            {"cat2": 3},
            [3],
        ),
        (
            "simple_df",
            ["nonexistent"],
            None,
            ValueError("Prescribed features {'nonexistent'} not found in dataset columns"),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ),
        (
            "simple_df",
            None,
            ["nonexistent"],
            ValueError("Prescribed features {'nonexistent'} not found in dataset columns"),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ),
        (
            "complex_df",
            None,
            None,
            None,
            ["int_col", "float_col", "str_col", "bool_col", "date_col", "cat_col"],
            6,
            ["int_col", "float_col", "bool_col"],
            3,
            {"int_col": np.int64, "float_col": np.float64, "bool_col": np.bool},
            ["str_col", "date_col", "cat_col"],
            3,
            {
                "str_col": np.object_,
                "date_col": np.datetime64,
                "cat_col": pd.CategoricalDtype.type,
            },
            {
                "str_col": ["A", "B", "C", "D", "E"],
                "date_col": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"],
                "cat_col": ["X", "Y", "Z"],
            },
            {"str_col": 5, "date_col": 5, "cat_col": 3},
            [5, 5, 3],
        ),
    ],
    indirect=["df_dispatcher"],
)
def test_fit(
    df_dispatcher: pd.DataFrame,
    ls_cts_features: Optional[List[str]],
    ls_cat_features: Optional[List[str]],
    expected_exception: Optional[BaseException],
    expected_ls_features: Optional[List[str]],
    expected_n_features: Optional[int],
    expected_ls_cts_features: Optional[List[str]],
    expected_n_cts_features: Optional[int],
    expected_dict_cts_dtypes: Optional[Dict[str, TabularDataDType]],
    expected_ls_cat_features: Optional[List[str]],
    expected_n_cat_features: Optional[int],
    expected_dict_cat_dtypes: Optional[Dict[str, TabularDataDType]],
    expected_cat_unique_map: Optional[Dict[str, List[str]]],
    expected_cat_unique_count_map: Optional[Dict[str, int]],
    expected_ls_n_cat: Optional[List[int]],
):
    spec = TabularDataSpec.build(
        ls_cts_features=ls_cts_features,
        ls_cat_features=ls_cat_features,
    )
    if expected_exception is not None:
        expected_exception_type = type(expected_exception)
        expected_exception_message = expected_exception.args[0]
        with pytest.raises(
            expected_exception_type,
            match=expected_exception_message,
        ):
            spec.fit(df=df_dispatcher)
    else:
        spec.fit(df=df_dispatcher)
        assert spec.ls_features == expected_ls_features, (
            f"Expected ls_features: {expected_ls_features} got {spec.ls_features}"
        )
        assert spec.n_features == expected_n_features, (
            f"Expected n_features: {expected_n_features} got {spec.n_features}"
        )
        assert spec.ls_cts_features == expected_ls_cts_features, (
            f"Expected ls_cts_features: {expected_ls_cts_features} got {spec.ls_cts_features}"
        )
        assert spec.n_cts_features == expected_n_cts_features, (
            f"Expected n_cts_features: {expected_n_cts_features} got {spec.n_cts_features}"
        )
        assert spec.dict_cts_dtypes == expected_dict_cts_dtypes, (
            f"Expected dict_cts_dtypes: {expected_dict_cts_dtypes} got {spec.dict_cts_dtypes}"
        )
        assert spec.ls_cat_features == expected_ls_cat_features, (
            f"Expected ls_cat_features: {expected_ls_cat_features} got {spec.ls_cat_features}"
        )
        assert spec.n_cat_features == expected_n_cat_features, (
            f"Expected n_cat_features: {expected_n_cat_features} got {spec.n_cat_features}"
        )
        assert spec.dict_cat_dtypes == expected_dict_cat_dtypes, (
            f"Expected dict_cat_dtypes: {expected_dict_cat_dtypes} got {spec.dict_cat_dtypes}"
        )
        assert spec.cat_unique_map == expected_cat_unique_map, (
            f"Expected cat_unique_map: {expected_cat_unique_map} got {spec.cat_unique_map}"
        )
        assert spec.cat_unique_count_map == expected_cat_unique_count_map, (
            f"Expected cat_unique_count_map: {expected_cat_unique_count_map}"
            + f"got {spec.cat_unique_count_map}"
        )
        assert spec.ls_n_cat == expected_ls_n_cat, (
            f"Expected ls_n_cat: {expected_ls_n_cat} got {spec.ls_n_cat}"
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "df_dispatcher, ls_cts_features, ls_cat_features, "
    + "expected_exception, "
    + "expected_ls_features, expected_n_features, "
    + "expected_ls_cts_features, expected_n_cts_features, expected_dict_cts_dtypes, "
    + "expected_ls_cat_features, expected_n_cat_features, expected_dict_cat_dtypes, "
    + "expected_cat_unique_map, expected_cat_unique_count_map, expected_ls_n_cat",
    [
        (
            "simple_df",
            None,
            None,
            None,
            ["num1", "num2", "cat1", "cat2"],
            4,
            ["num1", "num2"],
            2,
            {"num1": np.float64, "num2": np.float64},
            ["cat1", "cat2"],
            2,
            {"cat1": np.object_, "cat2": np.object_},
            {"cat1": ["A", "B", "C"], "cat2": ["X", "Y", "Z"]},
            {"cat1": 3, "cat2": 3},
            [3, 3],
        ),
        (
            "simple_df",
            ["num1"],
            None,
            None,
            ["num1", "num2", "cat1", "cat2"],
            4,
            ["num1", "num2"],
            2,
            {"num1": np.float64, "num2": np.float64},
            ["cat1", "cat2"],
            2,
            {"cat1": np.object_, "cat2": np.object_},
            {"cat1": ["A", "B", "C"], "cat2": ["X", "Y", "Z"]},
            {"cat1": 3, "cat2": 3},
            [3, 3],
        ),
        (
            "simple_df",
            None,
            ["cat1"],
            None,
            ["num1", "num2", "cat1", "cat2"],
            4,
            ["num1", "num2"],
            2,
            {"num1": np.float64, "num2": np.float64},
            ["cat1", "cat2"],
            2,
            {"cat1": np.object_, "cat2": np.object_},
            {"cat1": ["A", "B", "C"], "cat2": ["X", "Y", "Z"]},
            {"cat1": 3, "cat2": 3},
            [3, 3],
        ),
        (
            "simple_df",
            ["num1"],
            ["cat1"],
            None,
            ["num1", "num2", "cat1", "cat2"],
            4,
            ["num1", "num2"],
            2,
            {"num1": np.float64, "num2": np.float64},
            ["cat1", "cat2"],
            2,
            {"cat1": np.object_, "cat2": np.object_},
            {"cat1": ["A", "B", "C"], "cat2": ["X", "Y", "Z"]},
            {"cat1": 3, "cat2": 3},
            [3, 3],
        ),
        (
            "simple_df",
            None,
            ["num1"],
            None,
            ["num1", "num2", "cat1", "cat2"],
            4,
            ["num2"],
            1,
            {"num2": np.float64},
            ["num1", "cat1", "cat2"],
            3,
            {"num1": np.float64, "cat1": np.object_, "cat2": np.object_},
            {
                "num1": ["1.0", "2.0", "3.0", "4.0", "5.0"],
                "cat1": ["A", "B", "C"],
                "cat2": ["X", "Y", "Z"],
            },
            {"num1": 5, "cat1": 3, "cat2": 3},
            [5, 3, 3],
        ),
        (
            "simple_df",
            ["cat1"],
            None,
            None,
            ["num1", "num2", "cat1", "cat2"],
            4,
            ["num1", "num2", "cat1"],
            3,
            {
                "num1": np.float64,
                "num2": np.float64,
                "cat1": np.object_,
            },
            ["cat2"],
            1,
            {"cat2": np.object_},
            {"cat2": ["X", "Y", "Z"]},
            {"cat2": 3},
            [3],
        ),
        (
            "simple_df",
            ["num1", "cat1"],
            ["cat1"],
            ValueError("Categorical and continuous features overlap: {'cat1'}"),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ),
        (
            "simple_df",
            ["nonexistent"],
            None,
            ValueError("Prescribed features {'nonexistent'} not found in dataset columns"),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ),
        (
            "simple_df",
            None,
            ["nonexistent"],
            ValueError("Prescribed features {'nonexistent'} not found in dataset columns"),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ),
        (
            "complex_df",
            None,
            None,
            None,
            ["int_col", "float_col", "str_col", "bool_col", "date_col", "cat_col"],
            6,
            ["int_col", "float_col", "bool_col"],
            3,
            {"int_col": np.int64, "float_col": np.float64, "bool_col": np.bool},
            ["str_col", "date_col", "cat_col"],
            3,
            {
                "str_col": np.object_,
                "date_col": np.datetime64,
                "cat_col": pd.CategoricalDtype.type,
            },
            {
                "str_col": ["A", "B", "C", "D", "E"],
                "date_col": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"],
                "cat_col": ["X", "Y", "Z"],
            },
            {"str_col": 5, "date_col": 5, "cat_col": 3},
            [5, 5, 3],
        ),
    ],
    indirect=["df_dispatcher"],
)
def test_from_df(
    df_dispatcher: pd.DataFrame,
    ls_cts_features: Optional[List[str]],
    ls_cat_features: Optional[List[str]],
    expected_exception: Optional[BaseException],
    expected_ls_features: Optional[List[str]],
    expected_n_features: Optional[int],
    expected_ls_cts_features: Optional[List[str]],
    expected_n_cts_features: Optional[int],
    expected_dict_cts_dtypes: Optional[Dict[str, TabularDataDType]],
    expected_ls_cat_features: Optional[List[str]],
    expected_n_cat_features: Optional[int],
    expected_dict_cat_dtypes: Optional[Dict[str, TabularDataDType]],
    expected_cat_unique_map: Optional[Dict[str, List[str]]],
    expected_cat_unique_count_map: Optional[Dict[str, int]],
    expected_ls_n_cat: Optional[List[int]],
):
    if expected_exception is not None:
        expected_exception_type = type(expected_exception)
        expected_exception_message = expected_exception.args[0]
        with pytest.raises(
            expected_exception_type,
            match=expected_exception_message,
        ):
            _ = TabularDataSpec.from_df(
                df=df_dispatcher,
                ls_cts_features=ls_cts_features,
                ls_cat_features=ls_cat_features,
            )
    else:
        spec = TabularDataSpec.from_df(
            df=df_dispatcher,
            ls_cts_features=ls_cts_features,
            ls_cat_features=ls_cat_features,
        )
        assert spec.ls_features == expected_ls_features, (
            f"Expected ls_features: {expected_ls_features} got {spec.ls_features}"
        )
        assert spec.n_features == expected_n_features, (
            f"Expected n_features: {expected_n_features} got {spec.n_features}"
        )
        assert spec.ls_cts_features == expected_ls_cts_features, (
            f"Expected ls_cts_features: {expected_ls_cts_features} got {spec.ls_cts_features}"
        )
        assert spec.n_cts_features == expected_n_cts_features, (
            f"Expected n_cts_features: {expected_n_cts_features} got {spec.n_cts_features}"
        )
        assert spec.dict_cts_dtypes == expected_dict_cts_dtypes, (
            f"Expected dict_cts_dtypes: {expected_dict_cts_dtypes} got {spec.dict_cts_dtypes}"
        )
        assert spec.ls_cat_features == expected_ls_cat_features, (
            f"Expected ls_cat_features: {expected_ls_cat_features} got {spec.ls_cat_features}"
        )
        assert spec.n_cat_features == expected_n_cat_features, (
            f"Expected n_cat_features: {expected_n_cat_features} got {spec.n_cat_features}"
        )
        assert spec.dict_cat_dtypes == expected_dict_cat_dtypes, (
            f"Expected dict_cat_dtypes: {expected_dict_cat_dtypes} got {spec.dict_cat_dtypes}"
        )
        assert spec.cat_unique_map == expected_cat_unique_map, (
            f"Expected cat_unique_map: {expected_cat_unique_map} got {spec.cat_unique_map}"
        )
        assert spec.cat_unique_count_map == expected_cat_unique_count_map, (
            f"Expected cat_unique_count_map: {expected_cat_unique_count_map}"
            + f"got {spec.cat_unique_count_map}"
        )
        assert spec.ls_n_cat == expected_ls_n_cat, (
            f"Expected ls_n_cat: {expected_ls_n_cat} got {spec.ls_n_cat}"
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "df_dispatcher, ls_cts_features, ls_cat_features, feature_name, "
    + "expected_exception, expected_categories",
    [
        (
            "simple_df",
            None,
            None,
            "cat1",
            None,
            ["A", "B", "C"],
        ),
        (
            "complex_df",
            None,
            None,
            "str_col",
            None,
            ["A", "B", "C", "D", "E"],
        ),
        (
            "simple_df",
            None,
            None,
            "num1",
            ValueError("Feature 'num1' is not a categorical feature."),
            None,
        ),
        (
            "simple_df",
            None,
            None,
            "nonexistent",
            ValueError("Feature 'nonexistent' is not a categorical feature."),
            None,
        ),
    ],
    indirect=["df_dispatcher"],
)
def test_get_unique_categories(
    df_dispatcher: pd.DataFrame,
    ls_cts_features: Optional[List[str]],
    ls_cat_features: Optional[List[str]],
    feature_name: str,
    expected_exception: BaseException,
    expected_categories: Optional[List[str]],
):
    spec = TabularDataSpec.from_df(
        df=df_dispatcher,
        ls_cts_features=ls_cts_features,
        ls_cat_features=ls_cat_features,
    )
    if expected_exception is not None:
        expected_exception_type = type(expected_exception)
        expected_exception_message = expected_exception.args[0]
        with pytest.raises(
            expected_exception_type,
            match=expected_exception_message,
        ):
            spec.get_unique_categories(feature_name)
    else:
        ls_unique_categories = spec.get_unique_categories(feature_name)
        assert ls_unique_categories == expected_categories, (
            f"Expected unique categories for {feature_name}: {expected_categories} "
            f"got {ls_unique_categories}"
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "df_dispatcher, ls_cts_features, ls_cat_features, feature_name, "
    + "expected_exception, expected_count",
    [
        (
            "simple_df",
            None,
            None,
            "cat1",
            None,
            3,
        ),
        (
            "complex_df",
            None,
            None,
            "str_col",
            None,
            5,
        ),
        (
            "simple_df",
            None,
            None,
            "num1",
            ValueError("Feature 'num1' is not a categorical feature."),
            None,
        ),
        (
            "simple_df",
            None,
            None,
            "nonexistent",
            ValueError("Feature 'nonexistent' is not a categorical feature."),
            None,
        ),
    ],
    indirect=["df_dispatcher"],
)
def test_get_n_unique_categories(
    df_dispatcher: pd.DataFrame,
    ls_cts_features: Optional[List[str]],
    ls_cat_features: Optional[List[str]],
    feature_name: str,
    expected_exception: BaseException,
    expected_count: Optional[int],
):
    spec = TabularDataSpec.from_df(
        df=df_dispatcher,
        ls_cts_features=ls_cts_features,
        ls_cat_features=ls_cat_features,
    )
    if expected_exception is not None:
        expected_exception_type = type(expected_exception)
        expected_exception_message = expected_exception.args[0]
        with pytest.raises(
            expected_exception_type,
            match=expected_exception_message,
        ):
            spec.get_unique_categories(feature_name)
    else:
        n_unique_categories = spec.get_n_unique_categories(feature_name)
        assert n_unique_categories == expected_count, (
            f"Expected unique count for {feature_name}: {expected_count} got {n_unique_categories}"
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "df_dispatcher, ls_cts_features, ls_cat_features",
    [
        (
            "simple_df",
            None,
            None,
        ),
        (
            "complex_df",
            None,
            None,
        ),
    ],
    indirect=["df_dispatcher"],
)
def test_serialization_deserialization(
    df_dispatcher: pd.DataFrame,
    ls_cts_features: Optional[List[str]],
    ls_cat_features: Optional[List[str]],
):
    spec = TabularDataSpec.from_df(
        df=df_dispatcher,
        ls_cts_features=ls_cts_features,
        ls_cat_features=ls_cat_features,
    )
    json_str = spec.serialize()
    assert isinstance(json_str, str)
    deserialized_spec = TabularDataSpec.deserialize(json_str=json_str)
    assert deserialized_spec.ls_features == spec.ls_features
    assert deserialized_spec.n_features == spec.n_features
    assert deserialized_spec.ls_cts_features == spec.ls_cts_features
    assert deserialized_spec.n_cts_features == spec.n_cts_features
    assert set(deserialized_spec.dict_cts_dtypes.keys()) == set(spec.dict_cts_dtypes.keys())
    for key in spec.dict_cts_dtypes:
        assert deserialized_spec.dict_cts_dtypes[key] == spec.dict_cts_dtypes[key]
    assert deserialized_spec.ls_cat_features == spec.ls_cat_features
    assert deserialized_spec.n_cat_features == spec.n_cat_features
    assert set(deserialized_spec.dict_cat_dtypes.keys()) == set(spec.dict_cat_dtypes.keys())
    for key in spec.dict_cat_dtypes:
        assert deserialized_spec.dict_cat_dtypes[key] == spec.dict_cat_dtypes[key]

    assert deserialized_spec.cat_unique_map == spec.cat_unique_map
    assert deserialized_spec.cat_unique_count_map == spec.cat_unique_count_map
    assert deserialized_spec.ls_n_cat == spec.ls_n_cat


@pytest.mark.unit
@pytest.mark.parametrize(
    "df_dispatcher, ls_cts_features, ls_cat_features",
    [
        (
            "simple_df",
            None,
            None,
        ),
        (
            "complex_df",
            None,
            None,
        ),
    ],
    indirect=["df_dispatcher"],
)
def test_export_load(
    df_dispatcher: pd.DataFrame,
    ls_cts_features: Optional[List[str]],
    ls_cat_features: Optional[List[str]],
):
    spec = TabularDataSpec.from_df(
        df=df_dispatcher,
        ls_cts_features=ls_cts_features,
        ls_cat_features=ls_cat_features,
    )
    with TemporaryDirectory() as temp_dir:
        filepath = Path(temp_dir) / "spec.json"
        spec.export(filepath=filepath)
        assert filepath.exists()
        loaded_spec = TabularDataSpec.load(filepath=filepath)
        assert loaded_spec.ls_features == spec.ls_features
        assert loaded_spec.n_features == spec.n_features
        assert loaded_spec.ls_cts_features == spec.ls_cts_features
        assert loaded_spec.n_cts_features == spec.n_cts_features
        assert set(loaded_spec.dict_cts_dtypes.keys()) == set(spec.dict_cts_dtypes.keys())
        for key in spec.dict_cts_dtypes:
            assert loaded_spec.dict_cts_dtypes[key] == spec.dict_cts_dtypes[key]

        assert loaded_spec.ls_cat_features == spec.ls_cat_features
        assert loaded_spec.n_cat_features == spec.n_cat_features
        assert set(loaded_spec.dict_cat_dtypes.keys()) == set(spec.dict_cat_dtypes.keys())
        for key in spec.dict_cat_dtypes:
            assert loaded_spec.dict_cat_dtypes[key] == spec.dict_cat_dtypes[key]

        assert loaded_spec.cat_unique_map == spec.cat_unique_map
        assert loaded_spec.cat_unique_count_map == spec.cat_unique_count_map
        assert loaded_spec.ls_n_cat == spec.ls_n_cat
