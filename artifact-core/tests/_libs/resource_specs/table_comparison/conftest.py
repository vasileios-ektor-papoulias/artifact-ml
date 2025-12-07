import pandas as pd
import pytest


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
