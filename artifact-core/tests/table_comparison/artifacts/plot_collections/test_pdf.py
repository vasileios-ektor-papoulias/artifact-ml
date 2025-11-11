from typing import Dict
from unittest.mock import ANY

import pandas as pd
from artifact_core._base.artifact_dependencies import NO_ARTIFACT_HYPERPARAMS
from artifact_core._libs.implementation.tabular.pdf.overlaid_plotter import (
    TabularOverlaidPDFPlotter,
)
from artifact_core._libs.resource_spec.tabular.protocol import TabularDataSpecProtocol
from artifact_core.table_comparison._artifacts.base import (
    DatasetComparisonArtifactResources,
)
from artifact_core.table_comparison._artifacts.plot_collections.pdf import (
    PDFPlots,
)
from matplotlib.figure import Figure
from pytest_mock import MockerFixture


def test_compute(
    mocker: MockerFixture,
    resource_spec: TabularDataSpecProtocol,
    df_real: pd.DataFrame,
    df_synthetic: pd.DataFrame,
):
    fake_plots: Dict[str, Figure] = {
        "cts_1": Figure(),
        "cts_2": Figure(),
        "cat_1": Figure(),
        "cat_2": Figure(),
    }
    patch_get_overlaid_pdf_plot_collection = mocker.patch.object(
        target=TabularOverlaidPDFPlotter,
        attribute="get_overlaid_pdf_plot_collection",
        return_value=fake_plots,
    )
    artifact = PDFPlots(resource_spec=resource_spec, hyperparams=NO_ARTIFACT_HYPERPARAMS)
    resources = DatasetComparisonArtifactResources(
        dataset_real=df_real, dataset_synthetic=df_synthetic
    )
    result = artifact.compute(resources=resources)
    patch_get_overlaid_pdf_plot_collection.assert_called_once_with(
        dataset_real=ANY,
        dataset_synthetic=ANY,
        ls_cts_features=resource_spec.ls_cts_features,
        ls_cat_features=resource_spec.ls_cat_features,
        ls_features_order=resource_spec.ls_features,
        cat_unique_map=resource_spec.cat_unique_map,
    )
    _, kwargs = patch_get_overlaid_pdf_plot_collection.call_args
    pd.testing.assert_frame_equal(kwargs["dataset_real"], df_real)
    pd.testing.assert_frame_equal(kwargs["dataset_synthetic"], df_synthetic)
    assert result == fake_plots
