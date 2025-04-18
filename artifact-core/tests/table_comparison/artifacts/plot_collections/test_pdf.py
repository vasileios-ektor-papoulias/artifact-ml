from types import SimpleNamespace
from typing import Callable, Dict, List, cast
from unittest.mock import ANY

import pandas as pd
import pytest
from artifact_core.libs.data_spec.tabular.protocol import TabularDataSpecProtocol
from artifact_core.libs.implementation.pdf.overlaid_plotter import OverlaidPDFPlotter
from artifact_core.table_comparison.artifacts.base import (
    DatasetComparisonArtifactResources,
)
from artifact_core.table_comparison.artifacts.plot_collections.pdf import (
    PDFComparisonPlotCollection,
)
from matplotlib.figure import Figure
from pytest_mock import MockerFixture


@pytest.fixture
def tabular_spec_factory():
    def _factory(
        ls_cts_features: List[str], ls_cat_features: List[str], cat_unique_map: Dict[str, List[str]]
    ):
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
            cat_unique_map=cat_unique_map,
            cat_unique_count_map={
                feat: len(ls_unique) for feat, ls_unique in cat_unique_map.items()
            },
        )
        return cast(TabularDataSpecProtocol, spec)

    return _factory


@pytest.fixture
def df_real():
    return pd.DataFrame({"c1": [0.1, 0.4, 0.6], "c2": [1.2, 1.5, 1.8], "c3": ["A", "B", "B"]})


@pytest.fixture
def df_synth():
    return pd.DataFrame({"c1": [0.2, 0.5, 0.7], "c2": [1.0, 1.3, 2.0], "c3": ["A", "A", "A"]})


def test_compute(
    mocker: MockerFixture,
    tabular_spec_factory: Callable[
        [List[str], List[str], Dict[str, List[str]]], TabularDataSpecProtocol
    ],
    df_real: pd.DataFrame,
    df_synth: pd.DataFrame,
):
    spec = tabular_spec_factory(["c1", "c2"], ["c3"], {"c3": ["A", "B"]})
    fake_plots: Dict[str, Figure] = {
        "c1": Figure(),
        "c2": Figure(),
    }
    patcher = mocker.patch.object(
        OverlaidPDFPlotter,
        "get_overlaid_pdf_plot_collection",
        return_value=fake_plots,
    )
    artifact = PDFComparisonPlotCollection(data_spec=spec)
    resources = DatasetComparisonArtifactResources(dataset_real=df_real, dataset_synthetic=df_synth)
    result = artifact(resources=resources)
    patcher.assert_called_once_with(
        dataset_real=ANY,
        dataset_synthetic=ANY,
        ls_cts_features=spec.ls_cts_features,
        ls_cat_features=spec.ls_cat_features,
        ls_features_order=spec.ls_features,
        cat_unique_map=spec.cat_unique_map,
    )
    _, kwargs = patcher.call_args
    pd.testing.assert_frame_equal(kwargs["dataset_real"], df_real)
    pd.testing.assert_frame_equal(kwargs["dataset_synthetic"], df_synth)
    assert result == fake_plots
