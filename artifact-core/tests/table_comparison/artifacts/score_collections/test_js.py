from typing import Dict
from unittest.mock import ANY

import pandas as pd
import pytest
from artifact_core.libs.implementation.tabular.js.js import JSDistanceCalculator
from artifact_core.libs.resource_spec.tabular.protocol import TabularDataSpecProtocol
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


def test_compute(
    mocker: MockerFixture,
    resource_spec: TabularDataSpecProtocol,
    df_real: pd.DataFrame,
    df_synthetic: pd.DataFrame,
    hyperparams: JSDistanceHyperparams,
):
    fake_scores: Dict[str, float] = {"c1": 0.123, "cat": 0.456}
    patch_compute_dict_js = mocker.patch.object(
        target=JSDistanceCalculator,
        attribute="compute_dict_js",
        return_value=fake_scores,
    )
    artifact = JSDistance(resource_spec=resource_spec, hyperparams=hyperparams)
    resources = DatasetComparisonArtifactResources(
        dataset_real=df_real, dataset_synthetic=df_synthetic
    )
    result = artifact.compute(resources=resources)
    patch_compute_dict_js.assert_called_once_with(
        df_real=ANY,
        df_synthetic=ANY,
        ls_cts_features=resource_spec.ls_cts_features,
        ls_cat_features=resource_spec.ls_cat_features,
        cat_unique_map=resource_spec.cat_unique_map,
        n_bins_cts_histogram=hyperparams.n_bins_cts_histogram,
        categorical_only=hyperparams.categorical_only,
    )
    _, kwargs = patch_compute_dict_js.call_args
    pd.testing.assert_frame_equal(kwargs["df_real"], df_real)
    pd.testing.assert_frame_equal(kwargs["df_synthetic"], df_synthetic)
    assert result == fake_scores
