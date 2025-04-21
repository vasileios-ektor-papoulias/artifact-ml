import pandas as pd
import pytest


@pytest.fixture
def df_simple() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cts_1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "cts_2": [5.0, 4.0, 3.0, 2.0, 1.0],
            "cat_1": ["A", "B", "A", "C", "B"],
            "cat_2": ["X", "Y", "X", "Z", "Y"],
        }
    )


@pytest.fixture
def df_complex() -> pd.DataFrame:
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
def df_mixed() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "num_with_nan": [1.0, 2.0, None, 4.0, 5.0],
            "num_with_str": [1.0, "two", 3.0, 4.0, 5.0],
            "cat_with_nan": ["A", "B", None, "C", "B"],
            "cat_with_num": ["X", 2, "Z", 4, "Y"],
        }
    )


@pytest.fixture
def df_no_rows() -> pd.DataFrame:
    return pd.DataFrame(columns=["cts_1", "cts_2", "cat_1", "cat_2"])


@pytest.fixture
def df_empty() -> pd.DataFrame:
    return pd.DataFrame()


@pytest.fixture
def df_dispatcher(request: pytest.FixtureRequest) -> pd.DataFrame:
    ls_df_fixture_names = ["df_simple", "df_complex", "df_mixed", "df_no_rows", "df_empty"]
    if request.param not in ls_df_fixture_names:
        raise ValueError(
            f"Data fixture {request.param} not found: "
            + f"fixture param should be one of: {ls_df_fixture_names}"
        )
    return request.getfixturevalue(request.param)
