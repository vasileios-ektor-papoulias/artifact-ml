from unittest.mock import ANY

import pandas as pd
import pytest
from artifact_core._libs.artifacts.table_comparison.js.calculator import JSDistanceCalculator
from artifact_core._libs.resource_specs.table_comparison.protocol import TabularDataSpecProtocol
from artifact_core.table_comparison._artifacts.base import DatasetComparisonArtifactResources
from artifact_core.table_comparison._artifacts.scores.mean_js import (
    MeanJSDistanceScore,
    MeanJSDistanceScoreHyperparams,
)
from pytest_mock import MockerFixture


@pytest.fixture
def hyperparams() -> MeanJSDistanceScoreHyperparams:
    return MeanJSDistanceScoreHyperparams(n_bins_cts_histogram=8, categorical_only=True)


def test_compute(
    mocker: MockerFixture,
    resource_spec: TabularDataSpecProtocol,
    df_real: pd.DataFrame,
    df_synthetic: pd.DataFrame,
    hyperparams: MeanJSDistanceScoreHyperparams,
):
    fake_score: float = 0.789
    patch_compute_mean_js = mocker.patch.object(
        target=JSDistanceCalculator,
        attribute="compute_mean_js",
        return_value=fake_score,
    )
    artifact = MeanJSDistanceScore(resource_spec=resource_spec, hyperparams=hyperparams)
    resources = DatasetComparisonArtifactResources(
        dataset_real=df_real, dataset_synthetic=df_synthetic
    )
    result = artifact.compute(resources=resources)
    patch_compute_mean_js.assert_called_once_with(
        df_real=ANY,
        df_synthetic=ANY,
        cts_features=resource_spec.cts_features,
        cat_features=resource_spec.cat_features,
        cat_unique_map=resource_spec.cat_unique_map,
        n_bins_cts_histogram=hyperparams.n_bins_cts_histogram,
        categorical_only=hyperparams.categorical_only,
    )
    _, kwargs = patch_compute_mean_js.call_args
    pd.testing.assert_frame_equal(kwargs["df_real"], df_real)
    pd.testing.assert_frame_equal(kwargs["df_synthetic"], df_synthetic)
    assert result == fake_score
