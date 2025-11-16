from unittest.mock import ANY

import pandas as pd
import pytest
from artifact_core._libs.artifacts.table_comparison.projections.truncated_svd import (
    TruncatedSVDHyperparams,
    TruncatedSVDProjector,
)
from artifact_core._libs.resources_spec.tabular.protocol import TabularDataSpecProtocol
from artifact_core.table_comparison._artifacts.base import DatasetComparisonArtifactResources
from artifact_core.table_comparison._artifacts.plots.truncated_svd import (
    TruncatedSVDJuxtapositionPlot,
    TruncatedSVDJuxtapositionPlotHyperparams,
)
from pytest_mock import MockerFixture


@pytest.fixture
def hyperparams() -> TruncatedSVDJuxtapositionPlotHyperparams:
    hyperparams = TruncatedSVDJuxtapositionPlotHyperparams(use_categorical=True)
    return hyperparams


def test_compute(
    mocker: MockerFixture,
    resource_spec: TabularDataSpecProtocol,
    df_real: pd.DataFrame,
    df_synthetic: pd.DataFrame,
    hyperparams: TruncatedSVDJuxtapositionPlotHyperparams,
):
    fake_fig = Figure()
    mock_proj = mocker.Mock()
    mock_proj.produce_projection_comparison_plot.return_value = fake_fig
    patch_build = mocker.patch.object(
        target=TruncatedSVDProjector,
        attribute="build",
        return_value=mock_proj,
    )
    artifact = TruncatedSVDJuxtapositionPlot(
        resource_spec=resource_spec,
        hyperparams=hyperparams,
    )
    resources = DatasetComparisonArtifactResources(
        dataset_real=df_real,
        dataset_synthetic=df_synthetic,
    )
    result = artifact.compute(resources=resources)
    patch_build.assert_called_once_with(
        ls_cat_features=resource_spec.ls_cat_features,
        ls_cts_features=resource_spec.ls_cts_features,
        projector_config=ANY,
    )
    _, build_kwargs = patch_build.call_args
    assert isinstance(build_kwargs["projector_config"], TruncatedSVDHyperparams)
    assert build_kwargs["projector_config"].use_categorical == hyperparams.use_categorical
    mock_proj.produce_projection_comparison_plot.assert_called_once()
    _, plot_kwargs = mock_proj.produce_projection_comparison_plot.call_args
    pd.testing.assert_frame_equal(plot_kwargs["dataset_real"], df_real)
    pd.testing.assert_frame_equal(plot_kwargs["dataset_synthetic"], df_synthetic)
    assert result is fake_fig
