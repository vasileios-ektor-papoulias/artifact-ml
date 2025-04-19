from typing import Dict
from unittest.mock import ANY

import pandas as pd
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


def test_call(
    mocker: MockerFixture,
    data_spec: TabularDataSpecProtocol,
    df_real: pd.DataFrame,
    df_synth: pd.DataFrame,
):
    fake_plots: Dict[str, Figure] = {
        "cts_1": Figure(),
        "cts_2": Figure(),
        "cat_1": Figure(),
        "cat_2": Figure(),
    }
    patcher = mocker.patch.object(
        OverlaidPDFPlotter,
        "get_overlaid_pdf_plot_collection",
        return_value=fake_plots,
    )
    artifact = PDFComparisonPlotCollection(data_spec=data_spec)
    resources = DatasetComparisonArtifactResources(dataset_real=df_real, dataset_synthetic=df_synth)
    result = artifact(resources=resources)
    patcher.assert_called_once_with(
        dataset_real=ANY,
        dataset_synthetic=ANY,
        ls_cts_features=data_spec.ls_cts_features,
        ls_cat_features=data_spec.ls_cat_features,
        ls_features_order=data_spec.ls_features,
        cat_unique_map=data_spec.cat_unique_map,
    )
    _, kwargs = patcher.call_args
    pd.testing.assert_frame_equal(kwargs["dataset_real"], df_real)
    pd.testing.assert_frame_equal(kwargs["dataset_synthetic"], df_synth)
    assert result == fake_plots
