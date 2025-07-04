from unittest.mock import ANY

import pandas as pd
from artifact_core.base.artifact_dependencies import NO_ARTIFACT_HYPERPARAMS
from artifact_core.libs.implementation.tabular.pdf.overlaid_plotter import OverlaidPDFPlotter
from artifact_core.libs.resource_spec.tabular.protocol import TabularDataSpecProtocol
from artifact_core.table_comparison.artifacts.base import DatasetComparisonArtifactResources
from artifact_core.table_comparison.artifacts.plots.pdf import PDFPlot
from matplotlib.figure import Figure
from pytest_mock import MockerFixture


def test_compute(
    mocker: MockerFixture,
    resource_spec: TabularDataSpecProtocol,
    df_real: pd.DataFrame,
    df_synthetic: pd.DataFrame,
):
    fake_fig = Figure()
    patch_get = mocker.patch.object(
        target=OverlaidPDFPlotter,
        attribute="get_overlaid_pdf_plot",
        return_value=fake_fig,
    )
    artifact = PDFPlot(
        resource_spec=resource_spec,
        hyperparams=NO_ARTIFACT_HYPERPARAMS,
    )
    resources = DatasetComparisonArtifactResources(
        dataset_real=df_real,
        dataset_synthetic=df_synthetic,
    )
    result = artifact.compute(resources=resources)
    patch_get.assert_called_once_with(
        dataset_real=ANY,
        dataset_synthetic=ANY,
        ls_features_order=resource_spec.ls_features,
        ls_cts_features=resource_spec.ls_cts_features,
        ls_cat_features=resource_spec.ls_cat_features,
        cat_unique_map=resource_spec.cat_unique_map,
    )
    _, kwargs = patch_get.call_args
    pd.testing.assert_frame_equal(kwargs["dataset_real"], df_real)
    pd.testing.assert_frame_equal(kwargs["dataset_synthetic"], df_synthetic)
    assert result is fake_fig
