from unittest.mock import ANY

import pandas as pd
import pytest
from artifact_core._libs.implementation.tabular.correlations.calculator import (
    CategoricalAssociationType,
    ContinuousAssociationType,
    CorrelationCalculator,
)
from artifact_core._libs.resource_spec.tabular.protocol import TabularDataSpecProtocol
from artifact_core._libs.utils.calculators.vector_distance_calculator import VectorDistanceMetric
from artifact_core.table_comparison._artifacts.base import DatasetComparisonArtifactResources
from artifact_core.table_comparison._artifacts.scores.correlation import (
    CorrelationDistanceScore,
    CorrelationDistanceScoreHyperparams,
)
from pytest_mock import MockerFixture


@pytest.fixture
def hyperparams() -> CorrelationDistanceScoreHyperparams:
    return CorrelationDistanceScoreHyperparams(
        categorical_association_type=CategoricalAssociationType.THEILS_U,
        continuous_association_type=ContinuousAssociationType.PEARSON,
        vector_distance_metric=VectorDistanceMetric.L2,
    )


def test_compute(
    mocker: MockerFixture,
    resource_spec: TabularDataSpecProtocol,
    df_real: pd.DataFrame,
    df_synthetic: pd.DataFrame,
    hyperparams: CorrelationDistanceScoreHyperparams,
):
    fake_score: float = 0.314
    patch_compute_correlation_distance = mocker.patch.object(
        target=CorrelationCalculator,
        attribute="compute_correlation_distance",
        return_value=fake_score,
    )
    artifact = CorrelationDistanceScore(resource_spec=resource_spec, hyperparams=hyperparams)
    resources = DatasetComparisonArtifactResources(
        dataset_real=df_real, dataset_synthetic=df_synthetic
    )
    result = artifact.compute(resources=resources)
    patch_compute_correlation_distance.assert_called_once_with(
        categorical_correlation_type=hyperparams.categorical_association_type,
        continuous_correlation_type=hyperparams.continuous_association_type,
        distance_metric=hyperparams.vector_distance_metric,
        dataset_real=ANY,
        dataset_synthetic=ANY,
        ls_cat_features=resource_spec.ls_cat_features,
    )
    _, kwargs = patch_compute_correlation_distance.call_args
    pd.testing.assert_frame_equal(kwargs["dataset_real"], df_real)
    pd.testing.assert_frame_equal(kwargs["dataset_synthetic"], df_synthetic)
    assert result == fake_score
