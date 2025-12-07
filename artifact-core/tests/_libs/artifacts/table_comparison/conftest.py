from typing import Tuple

import pandas as pd
import pytest


@pytest.fixture
def df_simple() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cts_1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "cts_2": [-5.0, 4.0, 3.0, -2.0, 1.0],
            "cat_1": ["A", "B", "A", "C", "B"],
            "cat_2": ["X", "Y", "X", "Z", "X"],
        }
    )


@pytest.fixture
def df_complex() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cts_1": [1, 2, 3, 4, 5],
            "cts_2": [1.1, 2.2, 3.3, 4.4, 5.5],
            "cat_1": ["A", "B", "C", "D", "E"],
            "cat_2": [True, False, True, False, True],
            "cat_3": pd.date_range(start="2023-01-01", periods=5),
            "cat_4": pd.Categorical(["X", "Y", "Z", "X", "Y"]),
        }
    )


@pytest.fixture
def df_small_real() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cts_1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "cts_2": [5.0, 4.0, 3.0, 2.0, 1.0],
            "cat_1": ["A", "B", "A", "C", "B"],
            "cat_2": ["X", "Y", "X", "Z", "X"],
        }
    )


@pytest.fixture
def df_small_synthetic() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cts_1": [11.5, 2.5, 3.5, 0.5, 1.1],
            "cts_2": [4.5, 3.5, 2.5, 1.5, 0.11],
            "cat_1": ["B", "B", "B", "B", "A"],
            "cat_2": ["Y", "X", "Y", "Y", "X"],
        }
    )


@pytest.fixture
def df_large_real() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cts_1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "cts_2": [5.0, 4.0, 3.0, 2.0, 1.0, 0.5, 1.5, 2.5, 3.5, 4.5],
            "cat_1": ["A", "B", "A", "C", "B", "D", "A", "B", "C", "D"],
            "cat_2": ["X", "Y", "X", "Z", "Y", "Z", "Y", "X", "Z", "Y"],
        }
    )


@pytest.fixture
def df_large_synthetic() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cts_1": [1.2, 2.3, 3.4, 4.5, 5.6, 6.7, 7.8, 8.9, 9.0, 10.1],
            "cts_2": [4.9, 3.8, 2.7, 1.6, 0.5, 0.6, 1.7, 2.8, 3.9, 4.0],
            "cat_1": ["B", "A", "C", "B", "A", "C", "D", "A", "B", "C"],
            "cat_2": ["Y", "X", "Z", "Y", "X", "Y", "Z", "X", "Y", "Z"],
        }
    )


@pytest.fixture
def df_dispatcher(request: pytest.FixtureRequest) -> pd.DataFrame:
    ls_df_fixture_names = ["df_simple", "df_complex"]
    if request.param not in ls_df_fixture_names:
        raise ValueError(
            f"Data fixture {request.param} not found: "
            + f"fixture param should be one of: {ls_df_fixture_names}"
        )
    return request.getfixturevalue(request.param)


@pytest.fixture
def df_pair_dispatcher(request: pytest.FixtureRequest) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ls_df_fixture_names = [
        "df_small_real",
        "df_small_synthetic",
        "df_large_real",
        "df_large_synthetic",
    ]
    if request.param[0] not in ls_df_fixture_names or request.param[1] not in ls_df_fixture_names:
        raise ValueError(
            f"Data fixture {request.param} not found: "
            + f"fixture param should be one of: {ls_df_fixture_names}"
        )
    return tuple(request.getfixturevalue(name) for name in request.param)
