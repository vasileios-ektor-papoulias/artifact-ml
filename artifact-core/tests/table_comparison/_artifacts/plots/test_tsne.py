from unittest.mock import ANY

import pandas as pd
import pytest
from artifact_core._libs.artifacts.table_comparison.projections.tsne import (
    TSNEHyperparams,
    TSNEProjector,
)
from artifact_core._libs.resource_specs.table_comparison.protocol import TabularDataSpecProtocol
from artifact_core.table_comparison._artifacts.base import DatasetComparisonArtifactResources
from artifact_core.table_comparison._artifacts.plots.tsne import (
    TSNEJuxtapositionPlot,
    TSNEJuxtapositionPlotHyperparams,
)
from matplotlib.figure import Figure
from pytest_mock import MockerFixture


@pytest.fixture
def hyperparams() -> TSNEJuxtapositionPlotHyperparams:
    hyperparams = TSNEJuxtapositionPlotHyperparams(
        use_categorical=True, perplexity=30.0, learning_rate="auto", max_iter=1000
    )
    return hyperparams


def test_compute(
    mocker: MockerFixture,
    resource_spec: TabularDataSpecProtocol,
    df_real: pd.DataFrame,
    df_synthetic: pd.DataFrame,
    hyperparams: TSNEJuxtapositionPlotHyperparams,
):
    fake_fig = Figure()
    mock_proj = mocker.Mock()
    mock_proj.produce_projection_comparison_plot.return_value = fake_fig
    patch_build = mocker.patch.object(
        target=TSNEProjector,
        attribute="build",
        return_value=mock_proj,
    )
    artifact = TSNEJuxtapositionPlot(
        resource_spec=resource_spec,
        hyperparams=hyperparams,
    )
    resources = DatasetComparisonArtifactResources(
        dataset_real=df_real,
        dataset_synthetic=df_synthetic,
    )
    result = artifact.compute(resources=resources)
    patch_build.assert_called_once_with(
        cat_features=resource_spec.cat_features,
        cts_features=resource_spec.cts_features,
        projector_config=ANY,
    )
    _, build_kwargs = patch_build.call_args
    assert isinstance(build_kwargs["projector_config"], TSNEHyperparams)
    assert build_kwargs["projector_config"].use_categorical == hyperparams.use_categorical
    assert build_kwargs["projector_config"].perplexity == hyperparams.perplexity
    assert build_kwargs["projector_config"].learning_rate == hyperparams.learning_rate
    assert build_kwargs["projector_config"].max_iter == hyperparams.max_iter
    mock_proj.produce_projection_comparison_plot.assert_called_once()
    _, plot_kwargs = mock_proj.produce_projection_comparison_plot.call_args
    pd.testing.assert_frame_equal(plot_kwargs["dataset_real"], df_real)
    pd.testing.assert_frame_equal(plot_kwargs["dataset_synthetic"], df_synthetic)
    assert result is fake_fig
