from unittest.mock import ANY

import pandas as pd
from artifact_core._base.core.hyperparams import NO_ARTIFACT_HYPERPARAMS
from artifact_core._base.typing.artifact_result import PlotCollection
from artifact_core._libs.artifacts.table_comparison.cdf.overlaid_plotter import (
    TabularOverlaidCDFPlotter,
)
from artifact_core._libs.resource_specs.table_comparison.protocol import TabularDataSpecProtocol
from artifact_core.table_comparison._artifacts.base import (
    DatasetComparisonArtifactResources,
)
from artifact_core.table_comparison._artifacts.plot_collections.cdf import (
    CDFPlots,
)
from matplotlib.figure import Figure
from pytest_mock import MockerFixture


def test_compute(
    mocker: MockerFixture,
    resource_spec: TabularDataSpecProtocol,
    df_real: pd.DataFrame,
    df_synthetic: pd.DataFrame,
):
    fake_plots: PlotCollection = {
        "cts_1": Figure(),
        "cts_2": Figure(),
    }
    patch_get_overlaid_plot_collection = mocker.patch.object(
        target=TabularOverlaidCDFPlotter,
        attribute="get_overlaid_cdf_plot_collection",
        return_value=fake_plots,
    )
    artifact = CDFPlots(resource_spec=resource_spec, hyperparams=NO_ARTIFACT_HYPERPARAMS)
    resources = DatasetComparisonArtifactResources(
        dataset_real=df_real, dataset_synthetic=df_synthetic
    )
    result = artifact.compute(resources=resources)
    patch_get_overlaid_plot_collection.assert_called_once_with(
        dataset_real=ANY,
        dataset_synthetic=ANY,
        cts_features=resource_spec.cts_features,
    )
    _, kwargs = patch_get_overlaid_plot_collection.call_args
    pd.testing.assert_frame_equal(kwargs["dataset_real"], df_real)
    pd.testing.assert_frame_equal(kwargs["dataset_synthetic"], df_synthetic)
    assert result == fake_plots
