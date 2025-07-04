from typing import Dict
from unittest.mock import ANY

import pandas as pd
import pytest
from artifact_core.libs.implementation.tabular.pairwise_correlation.plotter import (
    CategoricalAssociationType,
    ContinuousAssociationType,
    PairwiseCorrelationHeatmapPlotter,
)
from artifact_core.libs.resource_spec.tabular.protocol import TabularDataSpecProtocol
from artifact_core.table_comparison.artifacts.base import (
    DatasetComparisonArtifactResources,
)
from artifact_core.table_comparison.artifacts.plot_collections.correlations import (
    CorrelationPlotCollection,
    CorrelationPlotCollectionHyperparams,
)
from matplotlib.figure import Figure
from pytest_mock import MockerFixture


@pytest.fixture
def hyperparams() -> CorrelationPlotCollectionHyperparams:
    return CorrelationPlotCollectionHyperparams(
        categorical_association_type=CategoricalAssociationType.CRAMERS_V,
        continuous_association_type=ContinuousAssociationType.PEARSON,
    )


def test_compute(
    mocker: MockerFixture,
    resource_spec: TabularDataSpecProtocol,
    df_real: pd.DataFrame,
    df_synthetic: pd.DataFrame,
    hyperparams: CorrelationPlotCollectionHyperparams,
):
    fake_plots: Dict[str, Figure] = {
        "cts_1": Figure(),
        "cts_2": Figure(),
        "cat_1": Figure(),
        "cat_2": Figure(),
    }
    patch_get_correlation_plot_collection = mocker.patch.object(
        target=PairwiseCorrelationHeatmapPlotter,
        attribute="get_correlation_plot_collection",
        return_value=fake_plots,
    )
    artifact = CorrelationPlotCollection(resource_spec=resource_spec, hyperparams=hyperparams)
    resources = DatasetComparisonArtifactResources(
        dataset_real=df_real, dataset_synthetic=df_synthetic
    )
    result = artifact.compute(resources=resources)
    patch_get_correlation_plot_collection.assert_called_once_with(
        categorical_correlation_type=CategoricalAssociationType.CRAMERS_V,
        continuous_correlation_type=ContinuousAssociationType.PEARSON,
        dataset_real=ANY,
        dataset_synthetic=ANY,
        ls_cat_features=resource_spec.ls_cat_features,
    )
    _, kwargs = patch_get_correlation_plot_collection.call_args
    pd.testing.assert_frame_equal(kwargs["dataset_real"], df_real)
    pd.testing.assert_frame_equal(kwargs["dataset_synthetic"], df_synthetic)
    assert result == fake_plots
