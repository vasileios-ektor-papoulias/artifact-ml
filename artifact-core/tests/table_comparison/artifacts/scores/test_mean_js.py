from unittest.mock import ANY

import pandas as pd
import pytest
from artifact_core.libs.data_spec.tabular.protocol import TabularDataSpecProtocol
from artifact_core.libs.implementation.js.js import JSDistanceCalculator
from artifact_core.table_comparison.artifacts.base import DatasetComparisonArtifactResources
from artifact_core.table_comparison.artifacts.scores.mean_js import (
    MeanJSDistance,
    MeanJSDistanceHyperparams,
)
from pytest_mock import MockerFixture


@pytest.fixture
def hyperparams() -> MeanJSDistanceHyperparams:
    return MeanJSDistanceHyperparams(n_bins_cts_histogram=8, categorical_only=True)


def test_compute(
    mocker: MockerFixture,
    data_spec: TabularDataSpecProtocol,
    df_real: pd.DataFrame,
    df_synth: pd.DataFrame,
    hyperparams: MeanJSDistanceHyperparams,
):
    fake_score: float = 0.789
    patcher = mocker.patch.object(
        JSDistanceCalculator,
        "compute_mean_js",
        return_value=fake_score,
    )
    artifact = MeanJSDistance(data_spec=data_spec, hyperparams=hyperparams)
    resources = DatasetComparisonArtifactResources(dataset_real=df_real, dataset_synthetic=df_synth)
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
    pd.testing.assert_frame_equal(kwargs["df_synthetic"], df_synth)
    assert result == fake_score
