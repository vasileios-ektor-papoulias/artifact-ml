from re import escape
from typing import Dict, List, Optional

import pandas as pd
import pytest
from artifact_core._libs.resource_validation.tabular.table_validator import TableValidator


@pytest.mark.unit
@pytest.mark.parametrize(
    "df_dispatcher, ls_features, ls_cts_features, ls_cat_features, "
    + "expected_cols, expected_dtype_names, expected_exception",
    [
        (
            "df_simple",
            ["cts_1", "cts_2", "cat_1", "cat_2"],
            ["cts_1", "cts_2"],
            ["cat_1", "cat_2"],
            ["cts_1", "cts_2", "cat_1", "cat_2"],
            {"cts_1": "numeric", "cts_2": "numeric", "cat_1": "string", "cat_2": "string"},
            None,
        ),
        (
            "df_simple",
            ["cts_1", "cat_1"],
            ["cts_1"],
            ["cat_1"],
            ["cts_1", "cat_1"],
            {"cts_1": "numeric", "cat_1": "string"},
            None,
        ),
        (
            "df_complex",
            ["int_col", "float_col"],
            ["int_col", "float_col"],
            [],
            ["int_col", "float_col"],
            {"int_col": "numeric", "float_col": "numeric"},
            None,
        ),
        (
            "df_complex",
            ["str_col", "cat_col"],
            [],
            ["str_col", "cat_col"],
            ["str_col", "cat_col"],
            {"str_col": "string", "cat_col": "string"},
            None,
        ),
        (
            "df_complex",
            ["int_col", "str_col", "bool_col"],
            ["int_col", "bool_col"],
            ["str_col"],
            ["int_col", "str_col", "bool_col"],
            {"int_col": "numeric", "str_col": "string", "bool_col": "numeric"},
            None,
        ),
        (
            "df_mixed",
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
            "df_simple",
            ["cts_1", "missing_feature"],
            ["cts_1"],
            ["missing_feature"],
            ["cts_1"],
            {"cts_1": "numeric"},
            ValueError("Features ['missing_feature'] not found in DataFrame"),
        ),
        (
            "df_no_rows",
            ["cts_1", "cts_2", "cat_1", "cat_2"],
            ["cts_1", "cts_2"],
            ["cat_1", "cat_2"],
            [],
            {},
            ValueError("DataFrame must not be empty."),
        ),
        (
            "df_empty",
            [],
            [],
            [],
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
