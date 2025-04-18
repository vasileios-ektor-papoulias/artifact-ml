from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import pytest
from artifact_core.libs.data_spec.tabular.protocol import (
    TabularDataDType,
)
from artifact_core.libs.data_spec.tabular.spec import TabularDataSpec


@pytest.fixture
def simple_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "num1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "num2": [5.0, 4.0, 3.0, 2.0, 1.0],
            "cat1": ["A", "B", "A", "C", "B"],
            "cat2": ["X", "Y", "X", "Z", "Y"],
        }
    )


@pytest.fixture
def complex_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "int_col": [1, 2, 3, 4, 5],
            "float_col": [1.1, 2.2, 3.3, 4.4, 5.5],
            "str_col": ["A", "B", "C", "D", "E"],
            "bool_col": [True, False, True, False, True],
            "date_col": pd.date_range(start="2023-01-01", periods=5),
            "cat_col": pd.Categorical(["X", "Y", "Z", "X", "Y"]),
        }
    )


@pytest.fixture
def df_dispatcher(request: pytest.FixtureRequest) -> pd.DataFrame:
    ls_df_fixture_names = ["simple_df", "complex_df"]
    if request.param not in ls_df_fixture_names:
        raise ValueError(
            f"Data fixture not found: fixture param should be one of: {ls_df_fixture_names}"
        )
    return request.getfixturevalue(request.param)


@pytest.fixture
def spec_factory() -> Callable[
    [Optional[pd.DataFrame], Optional[List[str]], Optional[List[str]]], TabularDataSpec
]:
    def _factory(
        df: Optional[pd.DataFrame] = None,
        ls_cts_features: Optional[List[str]] = None,
        ls_cat_features: Optional[List[str]] = None,
    ) -> TabularDataSpec:
        if df is None:
            spec = TabularDataSpec.build()
        else:
            spec = TabularDataSpec.from_df(
                df=df,
                ls_cts_features=ls_cts_features,
                ls_cat_features=ls_cat_features,
            )
        return spec

    return _factory


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
            ["cts_1", "cts_2", "cts_3", "cat_1", "cat_2"],
            ["cat_1", "cat_2", "cat_3"],
            ValueError("Categorical and continuous features overlap: {'cat_1', 'cat_2'}"),
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
        # Test with complex_df, auto-detection
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


@pytest.mark.parametrize(
    "df_dispatcher, ls_cts_features, ls_cat_features, "
    + "new_cat_unique_map, expected_exception, "
    + "expected_cat_unique_map, expected_cat_unique_count_map, expected_ls_n_cat",
    [
        (
            "simple_df",
            None,
            None,
            {
                "cat1": ["A", "B", "C", "D"],
                "cat2": ["X", "Y", "Z", "W"],
            },
            None,
            {
                "cat1": ["A", "B", "C", "D"],
                "cat2": ["X", "Y", "Z", "W"],
            },
            {"cat1": 4, "cat2": 4},
            [4, 4],
        ),
        (
            "simple_df",
            None,
            None,
            {
                "cat1": ["A", "B", "C", "D"],
            },
            None,
            {
                "cat1": ["A", "B", "C", "D"],
                "cat2": ["X", "Y", "Z"],
            },
            {"cat1": 4, "cat2": 3},
            [4, 3],
        ),
        (
            "complex_df",
            None,
            None,
            {
                "str_col": ["A", "B", "C", "D", "E", "F"],
                "date_col": [
                    "2023-01-01",
                    "2023-01-02",
                    "2023-01-03",
                    "2023-01-04",
                    "2023-01-05",
                    "2023-01-06",
                ],
                "cat_col": ["X", "Y", "Z", "W"],
            },
            None,
            {
                "str_col": ["A", "B", "C", "D", "E", "F"],
                "date_col": [
                    "2023-01-01",
                    "2023-01-02",
                    "2023-01-03",
                    "2023-01-04",
                    "2023-01-05",
                    "2023-01-06",
                ],
                "cat_col": ["X", "Y", "Z", "W"],
            },
            {"str_col": 6, "date_col": 6, "cat_col": 4},
            [6, 6, 4],
        ),
        (
            "simple_df",
            None,
            None,
            "not a dict",
            ValueError("categorical_unique_map must be a dictionary."),
            None,
            None,
            None,
        ),
        (
            "simple_df",
            None,
            None,
            {"num1": ["A", "B", "C"]},
            ValueError("Feature 'num1' is not a categorical feature."),
            None,
            None,
            None,
        ),
        (
            "simple_df",
            None,
            None,
            {"cat1": "not a list"},
            ValueError("Unique categories for 'cat1' must be a list of strings or integers."),
            None,
            None,
            None,
        ),
    ],
    indirect=["df_dispatcher"],
)
def test_cat_unique_map_setter(
    df_dispatcher: pd.DataFrame,
    ls_cts_features: Optional[List[str]],
    ls_cat_features: Optional[List[str]],
    new_cat_unique_map: Dict[str, List[str]],
    expected_exception: BaseException,
    expected_cat_unique_map: Optional[Dict[str, List[str]]],
    expected_cat_unique_count_map: Optional[Dict[str, int]],
    expected_ls_n_cat: Optional[List[int]],
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
            spec.cat_unique_map = new_cat_unique_map
    else:
        spec.cat_unique_map = new_cat_unique_map
        assert spec.cat_unique_map == expected_cat_unique_map, (
            f"Expected cat_unique_map: {expected_cat_unique_map} got {spec.cat_unique_map}"
        )
        assert spec.cat_unique_count_map == expected_cat_unique_count_map, (
            f"Expected cat_unique_count_map: {expected_cat_unique_count_map} "
            f"got {spec.cat_unique_count_map}"
        )
        assert spec.ls_n_cat == expected_ls_n_cat, (
            f"Expected ls_n_cat: {expected_ls_n_cat} got {spec.ls_n_cat}"
        )


@pytest.mark.parametrize(
    "df_dispatcher, ls_cts_features, ls_cat_features, feature_name, new_categories, "
    + "expected_exception, expected_categories, expected_count",
    [
        (
            "simple_df",
            None,
            None,
            "cat1",
            ["A", "B", "C", "D", "E"],
            None,
            ["A", "B", "C", "D", "E"],
            5,
        ),
        (
            "simple_df",
            None,
            None,
            "cat2",
            ["X", "Y", "Z", "W"],
            None,
            ["X", "Y", "Z", "W"],
            4,
        ),
        (
            "complex_df",
            None,
            None,
            "str_col",
            ["A", "B", "C", "D", "E", "F", "G"],
            None,
            ["A", "B", "C", "D", "E", "F", "G"],
            7,
        ),
        (
            "simple_df",
            None,
            None,
            "num1",
            [],
            ValueError("Feature 'num1' is not a categorical feature."),
            None,
            None,
        ),
        (
            "simple_df",
            None,
            None,
            "nonexistent",
            [],
            ValueError("Feature 'nonexistent' is not a categorical feature."),
            None,
            None,
        ),
    ],
    indirect=["df_dispatcher"],
)
def test_set_unique_categories(
    df_dispatcher: pd.DataFrame,
    ls_cts_features: Optional[List[str]],
    ls_cat_features: Optional[List[str]],
    feature_name: str,
    new_categories: List[str],
    expected_exception: BaseException,
    expected_categories: Optional[List[str]],
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
            spec.set_unique_categories(feature_name=feature_name, unique_categories=new_categories)
    else:
        spec.set_unique_categories(feature_name=feature_name, unique_categories=new_categories)
        assert spec.get_unique_categories(feature_name) == expected_categories, (
            f"Expected unique categories for {feature_name}: {expected_categories} "
            f"got {spec.get_unique_categories(feature_name)}"
        )
        assert spec.get_num_unique_categories(feature_name) == expected_count, (
            f"Expected unique count for {feature_name}: {expected_count} "
            f"got {spec.get_num_unique_categories(feature_name)}"
        )
        assert spec.cat_unique_map[feature_name] == expected_categories, (
            f"Expected cat_unique_map[{feature_name}]: {expected_categories} "
            f"got {spec.cat_unique_map[feature_name]}"
        )
        assert spec.cat_unique_count_map[feature_name] == expected_count, (
            f"Expected cat_unique_count_map[{feature_name}]: {expected_count} "
            f"got {spec.cat_unique_count_map[feature_name]}"
        )


@pytest.mark.parametrize(
    "df_dispatcher, ls_cts_features, ls_cat_features",
    [
        # Test with simple_df
        (
            "simple_df",
            None,
            None,
        ),
        # Test with complex_df
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
    """Test serializing and deserializing a TabularDataSpec."""
    # Create a spec from the DataFrame
    spec = TabularDataSpec.from_df(
        df=df_dispatcher,
        ls_cts_features=ls_cts_features,
        ls_cat_features=ls_cat_features,
    )

    # Serialize the spec
    json_str = spec.serialize()
    assert isinstance(json_str, str)

    # Deserialize the spec
    deserialized_spec = TabularDataSpec.deserialize(json_str=json_str)

    # Check that the deserialized spec has the same properties
    assert deserialized_spec.ls_features == spec.ls_features
    assert deserialized_spec.n_features == spec.n_features
    assert deserialized_spec.ls_cts_features == spec.ls_cts_features
    assert deserialized_spec.n_cts_features == spec.n_cts_features
    assert deserialized_spec.dict_cts_dtypes == spec.dict_cts_dtypes
    assert deserialized_spec.ls_cat_features == spec.ls_cat_features
    assert deserialized_spec.n_cat_features == spec.n_cat_features
    assert deserialized_spec.dict_cat_dtypes == spec.dict_cat_dtypes
    assert deserialized_spec.cat_unique_map == spec.cat_unique_map
    assert deserialized_spec.cat_unique_count_map == spec.cat_unique_count_map
    assert deserialized_spec.ls_n_cat == spec.ls_n_cat


@pytest.mark.parametrize(
    "df_dispatcher, ls_cts_features, ls_cat_features",
    [
        # Test with simple_df
        (
            "simple_df",
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
        assert loaded_spec.dict_cts_dtypes == spec.dict_cts_dtypes
        assert loaded_spec.ls_cat_features == spec.ls_cat_features
        assert loaded_spec.n_cat_features == spec.n_cat_features
        assert loaded_spec.dict_cat_dtypes == spec.dict_cat_dtypes
        assert loaded_spec.cat_unique_map == spec.cat_unique_map
        assert loaded_spec.cat_unique_count_map == spec.cat_unique_count_map
        assert loaded_spec.ls_n_cat == spec.ls_n_cat
