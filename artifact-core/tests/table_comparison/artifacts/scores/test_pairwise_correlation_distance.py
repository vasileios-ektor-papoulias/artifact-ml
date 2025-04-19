from unittest.mock import ANY

import pandas as pd
import pytest
from artifact_core.libs.data_spec.tabular.protocol import TabularDataSpecProtocol
from artifact_core.libs.implementation.pairwsie_correlation.calculator import (
    CategoricalAssociationType,
    ContinuousAssociationType,
    PairwiseCorrelationCalculator,
)
from artifact_core.libs.utils.vector_distance_calculator import VectorDistanceMetric
from artifact_core.table_comparison.artifacts.base import DatasetComparisonArtifactResources
from artifact_core.table_comparison.artifacts.scores.pairwise_correlation_distance import (
    PairwiseCorrelationDistance,
    PairwiseCorrelationDistanceConfig,
)
from pytest_mock import MockerFixture


@pytest.fixture
def hyperparams() -> PairwiseCorrelationDistanceConfig:
    return PairwiseCorrelationDistanceConfig(
        categorical_association_type=CategoricalAssociationType.THEILS_U,
        continuous_association_type=ContinuousAssociationType.PEARSON,
        vector_distance_metric=VectorDistanceMetric.L2,
    )


def test_call(
    mocker: MockerFixture,
    data_spec: TabularDataSpecProtocol,
    df_real: pd.DataFrame,
    df_synth: pd.DataFrame,
    hyperparams: PairwiseCorrelationDistanceConfig,
):
    fake_score: float = 0.314
    patcher = mocker.patch.object(
        PairwiseCorrelationCalculator,
        "compute_correlation_distance",
        return_value=fake_score,
    )
    artifact = PairwiseCorrelationDistance(data_spec=data_spec, hyperparams=hyperparams)
    resources = DatasetComparisonArtifactResources(dataset_real=df_real, dataset_synthetic=df_synth)
    result = artifact(resources=resources)
    patcher.assert_called_once_with(
        categorical_correlation_type=hyperparams.categorical_association_type,
        continuous_correlation_type=hyperparams.continuous_association_type,
        distance_metric=hyperparams.vector_distance_metric,
        dataset_real=ANY,
        dataset_synthetic=ANY,
        ls_cat_features=data_spec.ls_cat_features,
    )
    _, kwargs = patcher.call_args
    pd.testing.assert_frame_equal(kwargs["dataset_real"], df_real)
    pd.testing.assert_frame_equal(kwargs["dataset_synthetic"], df_synth)
    assert result == fake_score
