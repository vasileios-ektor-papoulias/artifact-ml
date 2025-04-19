from unittest.mock import ANY

import pandas as pd
import pytest
from artifact_core.libs.data_spec.tabular.protocol import TabularDataSpecProtocol
from artifact_core.libs.implementation.projections.tsne import (
    TSNEHyperparams,
    TSNEProjector,
)
from artifact_core.table_comparison.artifacts.base import DatasetComparisonArtifactResources
from artifact_core.table_comparison.artifacts.plots.tsne_projection import (
    TSNEProjectionComparisonPlot,
    TSNEProjectionComparisonPlotConfig,
)
from matplotlib.figure import Figure
from pytest_mock import MockerFixture


@pytest.fixture
def hyperparams() -> TSNEProjectionComparisonPlotConfig:
    hyperparams = TSNEProjectionComparisonPlotConfig(
        use_categorical=True, perplexity=30.0, learning_rate="auto", n_iter=1000
    )
    return hyperparams


def test_call(
    mocker: MockerFixture,
    data_spec: TabularDataSpecProtocol,
    df_real: pd.DataFrame,
    df_synth: pd.DataFrame,
    hyperparams: TSNEProjectionComparisonPlotConfig,
):
    fake_fig = Figure()
    mock_proj = mocker.Mock()
    mock_proj.produce_projection_comparison_plot.return_value = fake_fig
    mock_build = mocker.patch.object(
        TSNEProjector,
        "build",
        return_value=mock_proj,
    )
    artifact = TSNEProjectionComparisonPlot(
        data_spec=data_spec,
        hyperparams=hyperparams,
    )
    resources = DatasetComparisonArtifactResources(
        dataset_real=df_real,
        dataset_synthetic=df_synth,
    )
    result = artifact(resources=resources)
    mock_build.assert_called_once_with(
        ls_cat_features=data_spec.ls_cat_features,
        ls_cts_features=data_spec.ls_cts_features,
        projector_config=ANY,
    )
    _, build_kwargs = mock_build.call_args
    assert isinstance(build_kwargs["projector_config"], TSNEHyperparams)
    assert build_kwargs["projector_config"].use_categorical == hyperparams.use_categorical
    assert build_kwargs["projector_config"].perplexity == hyperparams.perplexity
    assert build_kwargs["projector_config"].learning_rate == hyperparams.learning_rate
    assert build_kwargs["projector_config"].n_iter == hyperparams.n_iter
    mock_proj.produce_projection_comparison_plot.assert_called_once()
    _, plot_kwargs = mock_proj.produce_projection_comparison_plot.call_args
    pd.testing.assert_frame_equal(plot_kwargs["dataset_real"], df_real)
    pd.testing.assert_frame_equal(plot_kwargs["dataset_synthetic"], df_synth)
    assert result is fake_fig
