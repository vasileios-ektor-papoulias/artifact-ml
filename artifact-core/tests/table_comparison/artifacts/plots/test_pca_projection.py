from unittest.mock import ANY

import pandas as pd
import pytest
from artifact_core.libs.implementation.tabular.projections.pca import PCAHyperparams, PCAProjector
from artifact_core.libs.resource_spec.tabular.protocol import TabularDataSpecProtocol
from artifact_core.table_comparison.artifacts.base import (
    DatasetComparisonArtifactResources,
)
from artifact_core.table_comparison.artifacts.plots.pca_projection import (
    PCAProjectionComparisonPlot,
    PCAProjectionComparisonPlotConfig,
)
from matplotlib.figure import Figure
from pytest_mock import MockerFixture


@pytest.fixture
def hyperparams() -> PCAProjectionComparisonPlotConfig:
    hyperparams = PCAProjectionComparisonPlotConfig(use_categorical=True)
    return hyperparams


def test_compute(
    mocker: MockerFixture,
    resource_spec: TabularDataSpecProtocol,
    df_real: pd.DataFrame,
    df_synthetic: pd.DataFrame,
    hyperparams: PCAProjectionComparisonPlotConfig,
):
    fake_plot = Figure()
    mock_projector = mocker.Mock()
    mock_projector.produce_projection_comparison_plot.return_value = fake_plot
    patch_build = mocker.patch.object(
        target=PCAProjector, attribute="build", return_value=mock_projector
    )
    artifact = PCAProjectionComparisonPlot(
        resource_spec=resource_spec,
        hyperparams=hyperparams,
    )
    resources = DatasetComparisonArtifactResources(
        dataset_real=df_real, dataset_synthetic=df_synthetic
    )
    result = artifact.compute(resources=resources)
    patch_build.assert_called_once_with(
        ls_cat_features=resource_spec.ls_cat_features,
        ls_cts_features=resource_spec.ls_cts_features,
        projector_config=ANY,
    )
    _, kwargs = patch_build.call_args
    assert isinstance(kwargs["projector_config"], PCAHyperparams)
    assert kwargs["projector_config"].use_categorical == hyperparams.use_categorical
    mock_projector.produce_projection_comparison_plot.assert_called_once()
    _, kwargs = mock_projector.produce_projection_comparison_plot.call_args
    pd.testing.assert_frame_equal(kwargs["dataset_real"], df_real)
    pd.testing.assert_frame_equal(kwargs["dataset_synthetic"], df_synthetic)
    assert result == fake_plot
