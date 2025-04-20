from re import escape
from typing import Dict, List, Optional

import pandas as pd
import pytest
from artifact_core.libs.validation.table_validator import TableValidator


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
def mixed_df() -> pd.DataFrame:
    """DataFrame with mixed types and some problematic values."""
    return pd.DataFrame(
        {
            "num_with_nan": [1.0, 2.0, None, 4.0, 5.0],
            "num_with_str": [1.0, "two", 3.0, 4.0, 5.0],
            "cat_with_nan": ["A", "B", None, "C", "B"],
            "cat_with_num": ["X", 2, "Z", 4, "Y"],
        }
    )


@pytest.fixture
def empty_df() -> pd.DataFrame:
    """Empty DataFrame with columns."""
    return pd.DataFrame(columns=["num1", "num2", "cat1", "cat2"])


@pytest.fixture
def truly_empty_df() -> pd.DataFrame:
    """Empty DataFrame without columns."""
    return pd.DataFrame()


@pytest.fixture
def df_dispatcher(request: pytest.FixtureRequest) -> pd.DataFrame:
    ls_df_fixture_names = ["simple_df", "complex_df", "mixed_df", "empty_df", "truly_empty_df"]
    if request.param not in ls_df_fixture_names:
        raise ValueError(
            f"Data fixture not found: fixture param should be one of: {ls_df_fixture_names}"
        )
    return request.getfixturevalue(request.param)


@pytest.mark.parametrize(
    "df_dispatcher, ls_features, ls_cts_features, ls_cat_features, "
    + "expected_cols, expected_dtype_names, expected_exception",
    [
        (
            "simple_df",
            ["num1", "num2", "cat1", "cat2"],
            ["num1", "num2"],
            ["cat1", "cat2"],
            ["num1", "num2", "cat1", "cat2"],
            {"num1": "numeric", "num2": "numeric", "cat1": "string", "cat2": "string"},
            None,
        ),
        (
            "simple_df",
            ["num1", "cat1"],
            ["num1"],
            ["cat1"],
            ["num1", "cat1"],
            {"num1": "numeric", "cat1": "string"},
            None,
        ),
        (
            "complex_df",
            ["int_col", "float_col"],
            ["int_col", "float_col"],
            [],
            ["int_col", "float_col"],
            {"int_col": "numeric", "float_col": "numeric"},
            None,
        ),
        (
            "complex_df",
            ["str_col", "cat_col"],
            [],
            ["str_col", "cat_col"],
            ["str_col", "cat_col"],
            {"str_col": "string", "cat_col": "string"},
            None,
        ),
        (
            "complex_df",
            ["int_col", "str_col", "bool_col"],
            ["int_col", "bool_col"],
            ["str_col"],
            ["int_col", "str_col", "bool_col"],
            {"int_col": "numeric", "str_col": "string", "bool_col": "numeric"},
            None,
        ),
        (
            "mixed_df",
            ["num_with_nan", "num_with_str", "cat_with_nan", "cat_with_num"],
            ["num_with_nan", "num_with_str"],
            ["cat_with_nan", "cat_with_num"],
            ["num_with_nan", "num_with_str", "cat_with_nan", "cat_with_num"],
            {
                "num_with_nan": "numeric",
                "num_with_str": "numeric",
                "cat_with_nan": "string",
                "cat_with_num": "string",
            },
            None,
        ),
        (
            "simple_df",
            ["num1", "missing_feature"],
            ["num1"],
            ["missing_feature"],
            ["num1"],
            {"num1": "numeric"},
            ValueError("Features ['missing_feature'] not found in DataFrame"),
        ),
        (
            "empty_df",
            ["num1", "num2", "cat1", "cat2"],
            ["num1", "num2"],
            ["cat1", "cat2"],
            [],
            {},
            ValueError("DataFrame must not be empty."),
        ),
        (
            "truly_empty_df",
            ["col1", "col2"],
            ["col1"],
            ["col2"],
            [],
            {},
            ValueError("DataFrame must not be empty."),
        ),
    ],
    indirect=["df_dispatcher"],
)
def test_validate(
    df_dispatcher: pd.DataFrame,
    ls_features: List[str],
    ls_cts_features: List[str],
    ls_cat_features: List[str],
    expected_cols: List[str],
    expected_dtype_names: Dict[str, str],
    expected_exception: Optional[BaseException],
):
    if expected_exception is not None:
        expected_exception_type = type(expected_exception)
        expected_exception_message = expected_exception.args[0]
        with pytest.raises(expected_exception_type, match=escape(expected_exception_message)):
            TableValidator.validate(
                df=df_dispatcher,
                ls_features=ls_features,
                ls_cts_features=ls_cts_features,
                ls_cat_features=ls_cat_features,
            )
    else:
        result = TableValidator.validate(
            df=df_dispatcher,
            ls_features=ls_features,
            ls_cts_features=ls_cts_features,
            ls_cat_features=ls_cat_features,
        )
        assert set(result.columns) == set(expected_cols)
        for col, dtype in expected_dtype_names.items():
            if dtype == "numeric":
                assert pd.api.types.is_numeric_dtype(result[col]), f"Column {col} should be numeric"
            elif dtype == "string":
                non_nan_values = result[col].dropna()
                assert all(isinstance(val, str) for val in non_nan_values), (
                    f"Column {col} should contain strings"
                )
