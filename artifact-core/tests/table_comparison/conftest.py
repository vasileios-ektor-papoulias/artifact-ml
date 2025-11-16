from types import SimpleNamespace
from typing import cast

import pandas as pd
import pytest
from artifact_core._libs.resources_spec.tabular.protocol import TabularDataSpecProtocol


@pytest.fixture
def resource_spec() -> TabularDataSpecProtocol:
    ls_cts = ["cts_1", "cts_2"]
    ls_cat = ["cat_1", "cat_2"]
    ls_features = ls_cts + ls_cat
    cat_unique_map = {"cat_1": ["a", "b", "c"], "cat_2": ["A", "B", "C", "D"]}
    cat_unique_count_map = {feat: len(ls_unique) for feat, ls_unique in cat_unique_map.items()}
    spec = SimpleNamespace(
        ls_features=ls_features,
        n_features=len(ls_features),
        ls_cts_features=ls_cts,
        n_cts_features=len(ls_cts),
        dict_cts_dtypes={feat: float for feat in ls_cts},
        ls_cat_features=ls_cat,
        n_cat_features=len(ls_cat),
        dict_cat_dtypes={feat: str for feat in ls_cat},
        cat_unique_map=cat_unique_map,
        cat_unique_count_map=cat_unique_count_map,
    )
    return cast(TabularDataSpecProtocol, spec)


@pytest.fixture
def df_real() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cts_1": [1.0, 2.0, 3.0],
            "cts_2": [0.1, 0.2, 0.3],
            "cat_1": ["a", "b", "a"],
            "cat_2": ["A", "A", "C"],
        }
    )


@pytest.fixture
def df_synthetic() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cts_1": [4.0, 5.0, 6.0],
            "cts_2": [0.15, 0.21, 0.33],
            "cat_1": ["a", "a", "a"],
            "cat_2": ["B", "A", "D"],
        }
    )
