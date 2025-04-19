from unittest.mock import ANY

import pandas as pd
from artifact_core.libs.data_spec.tabular.protocol import TabularDataSpecProtocol
from artifact_core.libs.implementation.cdf.overlaid_plotter import OverlaidCDFPlotter
from artifact_core.table_comparison.artifacts.base import (
    DatasetComparisonArtifactResources,
)
from artifact_core.table_comparison.artifacts.plots.cdf import (
    CDFComparisonCombinedPlot,
)
from matplotlib.figure import Figure
from pytest_mock import MockerFixture


def test_compute(
    mocker: MockerFixture,
    data_spec: TabularDataSpecProtocol,
    df_real: pd.DataFrame,
    df_synthetic: pd.DataFrame,
):
    fake_plot = Figure()
    patch_get_overlaid_cdf_plot = mocker.patch.object(
        target=OverlaidCDFPlotter,
        attribute="get_overlaid_cdf_plot",
        return_value=fake_plot,
    )
    artifact = CDFComparisonCombinedPlot(data_spec=data_spec)
    resources = DatasetComparisonArtifactResources(
        dataset_real=df_real, dataset_synthetic=df_synthetic
    )
    result = artifact.compute(resources=resources)
    patch_get_overlaid_cdf_plot.assert_called_once_with(
        dataset_real=ANY,
        dataset_synthetic=ANY,
        ls_cts_features=data_spec.ls_cts_features,
    )
    _, kwargs = patch_get_overlaid_cdf_plot.call_args
    pd.testing.assert_frame_equal(kwargs["dataset_real"], df_real)
    pd.testing.assert_frame_equal(kwargs["dataset_synthetic"], df_synthetic)
    assert result == fake_plot
