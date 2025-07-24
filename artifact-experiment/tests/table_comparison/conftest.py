import pandas as pd
import pytest
from artifact_core.table_comparison import (
    TabularDataSpec,
)


@pytest.fixture
def tabular_data_spec() -> TabularDataSpec:
    return TabularDataSpec.build()


@pytest.fixture
def dataset_real_dispatcher(request) -> pd.DataFrame:
    return request.getfixturevalue(request.param)


@pytest.fixture
def dataset_synthetic_dispatcher(request) -> pd.DataFrame:
    return request.getfixturevalue(request.param)


@pytest.fixture
def df_1() -> pd.DataFrame:
    return pd.DataFrame({"a": [1, 2]})


@pytest.fixture
def df_2() -> pd.DataFrame:
    return pd.DataFrame({"a": [3, 4]})


@pytest.fixture
def df_3() -> pd.DataFrame:
    return pd.DataFrame({"x": [1.1, 2.2]})


@pytest.fixture
def df_4() -> pd.DataFrame:
    return pd.DataFrame({"x": [2.2, 3.3]})


@pytest.fixture
def df_5() -> pd.DataFrame:
    return pd.DataFrame()
