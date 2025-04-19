from typing import Dict
from unittest.mock import ANY

import pandas as pd
import pytest
from artifact_core.libs.data_spec.tabular.protocol import TabularDataSpecProtocol
from artifact_core.libs.implementation.js.js import JSDistanceCalculator
from artifact_core.table_comparison.artifacts.base import (
    DatasetComparisonArtifactResources,
)
from artifact_core.table_comparison.artifacts.score_collections.js import (
    JSDistance,
    JSDistanceHyperparams,
)
from pytest_mock import MockerFixture


@pytest.fixture
def hyperparams() -> JSDistanceHyperparams:
    return JSDistanceHyperparams(n_bins_cts_histogram=5, categorical_only=False)


def test_call(
    mocker: MockerFixture,
    data_spec: TabularDataSpecProtocol,
    df_real: pd.DataFrame,
    df_synthetic: pd.DataFrame,
    hyperparams: JSDistanceHyperparams,
):
    fake_scores: Dict[str, float] = {"c1": 0.123, "cat": 0.456}
    patcher = mocker.patch.object(
        JSDistanceCalculator,
        "compute_dict_js",
        return_value=fake_scores,
    )
    artifact = JSDistance(data_spec=data_spec, hyperparams=hyperparams)
    resources = DatasetComparisonArtifactResources(
        dataset_real=df_real, dataset_synthetic=df_synthetic
    )
    result = artifact(resources=resources)
    patcher.assert_called_once_with(
        df_real=ANY,
        df_synthetic=ANY,
        ls_cts_features=data_spec.ls_cts_features,
        ls_cat_features=data_spec.ls_cat_features,
        cat_unique_map=data_spec.cat_unique_map,
        n_bins_cts_histogram=hyperparams.n_bins_cts_histogram,
        categorical_only=hyperparams.categorical_only,
    )
    _, kwargs = patcher.call_args
    pd.testing.assert_frame_equal(kwargs["df_real"], df_real)
    pd.testing.assert_frame_equal(kwargs["df_synthetic"], df_synthetic)
    assert result == fake_scores
