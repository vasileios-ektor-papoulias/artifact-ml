from unittest.mock import ANY

import pandas as pd
import pytest
from artifact_core.libs.data_spec.tabular.protocol import TabularDataSpecProtocol
from artifact_core.libs.implementation.pairwsie_correlation.calculator import (
    CategoricalAssociationType,
    ContinuousAssociationType,
)
from artifact_core.libs.implementation.pairwsie_correlation.plotter import (
    PairwiseCorrelationHeatmapPlotter,
)
from artifact_core.table_comparison.artifacts.base import (
    DatasetComparisonArtifactResources,
)
from artifact_core.table_comparison.artifacts.plots.pairwise_correlations import (
    CorrelationComparisonCombinedPlot,
    CorrelationComparisonHeatmapConfig,
)
from matplotlib.figure import Figure
from pytest_mock import MockerFixture


@pytest.fixture
def hyperparams() -> CorrelationComparisonHeatmapConfig:
    hyperparams = CorrelationComparisonHeatmapConfig(
        categorical_association_type=CategoricalAssociationType.CRAMERS_V,
        continuous_association_type=ContinuousAssociationType.PEARSON,
    )
    return hyperparams


def test_compute(
    mocker: MockerFixture,
    data_spec: TabularDataSpecProtocol,
    df_real: pd.DataFrame,
    df_synthetic: pd.DataFrame,
    hyperparams: CorrelationComparisonHeatmapConfig,
):
    fake_plot = Figure()
    patch_get_combined_correlation_plot = mocker.patch.object(
        target=PairwiseCorrelationHeatmapPlotter,
        attribute="get_combined_correlation_plot",
        return_value=fake_plot,
    )
    artifact = CorrelationComparisonCombinedPlot(
        data_spec=data_spec,
        hyperparams=hyperparams,
    )
    resources = DatasetComparisonArtifactResources(
        dataset_real=df_real, dataset_synthetic=df_synthetic
    )

    result = artifact.compute(resources=resources)

    patch_get_combined_correlation_plot.assert_called_once_with(
        categorical_correlation_type=hyperparams.categorical_association_type,
        continuous_correlation_type=hyperparams.continuous_association_type,
        dataset_real=ANY,
        dataset_synthetic=ANY,
        ls_cat_features=data_spec.ls_cat_features,
    )

    _, kwargs = patch_get_combined_correlation_plot.call_args
    pd.testing.assert_frame_equal(kwargs["dataset_real"], df_real)
    pd.testing.assert_frame_equal(kwargs["dataset_synthetic"], df_synthetic)
    assert result == fake_plot
