from types import SimpleNamespace
from typing import Callable, Dict, List, cast
from unittest.mock import ANY

import pandas as pd
import pytest
from artifact_core.libs.data_spec.tabular.protocol import TabularDataSpecProtocol
from artifact_core.libs.implementation.cdf.overlaid_plotter import OverlaidCDFPlotter
from artifact_core.table_comparison.artifacts.base import (
    DatasetComparisonArtifactResources,
)
from artifact_core.table_comparison.artifacts.plot_collections.cdf import (
    CDFComparisonPlotCollection,
)
from matplotlib.figure import Figure
from pytest_mock import MockerFixture


@pytest.fixture
def tabular_spec_factory():
    def _factory(ls_cts_features: List[str], ls_cat_features: List[str]):
        ls_features = ls_cts_features + ls_cat_features
        spec = SimpleNamespace(
            ls_features=ls_features,
            n_features=len(ls_features),
            ls_cts_features=ls_cts_features,
            n_cts_features=len(ls_cts_features),
            dict_cts_dtypes={},
            ls_cat_features=ls_cat_features,
            n_cat_features=len(ls_cat_features),
            dict_cat_dtypes={},
            cat_unique_map={},
            cat_unique_count_map={},
        )
        return cast(TabularDataSpecProtocol, spec)

    return _factory


@pytest.fixture
def df_real():
    return pd.DataFrame({"c1": [0.1, 0.4, 0.6], "c2": [1.2, 1.5, 1.8]})


@pytest.fixture
def df_synth():
    return pd.DataFrame({"c1": [0.2, 0.5, 0.7], "c2": [1.0, 1.3, 2.0]})


def test_compute(
    mocker: MockerFixture,
    tabular_spec_factory: Callable[[List[str], List[str]], TabularDataSpecProtocol],
    df_real: pd.DataFrame,
    df_synth: pd.DataFrame,
):
    spec = tabular_spec_factory(["c1", "c2"], [])
    fake_plots: Dict[str, Figure] = {
        "c1": Figure(),
        "c2": Figure(),
    }
    patcher = mocker.patch.object(
        OverlaidCDFPlotter,
        "get_overlaid_cdf_plot_collection",
        return_value=fake_plots,
    )
    artifact = CDFComparisonPlotCollection(data_spec=spec)
    resources = DatasetComparisonArtifactResources(dataset_real=df_real, dataset_synthetic=df_synth)
    result = artifact(resources=resources)
    patcher.assert_called_once_with(
        dataset_real=ANY,
        dataset_synthetic=ANY,
        ls_cts_features=spec.ls_cts_features,
    )
    _, kwargs = patcher.call_args
    pd.testing.assert_frame_equal(kwargs["dataset_real"], df_real)
    pd.testing.assert_frame_equal(kwargs["dataset_synthetic"], df_synth)
    assert result == fake_plots
