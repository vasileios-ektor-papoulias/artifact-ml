from math import isclose
from typing import List, Optional, Tuple

import pandas as pd
import pytest
from artifact_core._libs.artifacts.table_comparison.correlations.calculator import (  # noqa: E501
    CategoricalAssociationType,
    ContinuousAssociationType,
    CorrelationCalculator,
)
from artifact_core._libs.tools.calculators.vector_distance_calculator import (
    VectorDistanceCalculator,
    VectorDistanceMetric,
)
from pytest_mock import MockerFixture


@pytest.mark.unit
@pytest.mark.parametrize(
    "df, cat_features, cat_corr, cont_corr, expected_shape",
    [
        (
            pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}),
            [],
            CategoricalAssociationType.CRAMERS_V,
            ContinuousAssociationType.PEARSON,
            (2, 2),
        ),
        (
            pd.DataFrame({"cat": ["a", "b", "a"], "num": [1, 2, 3]}),
            ["cat"],
            CategoricalAssociationType.THEILS_U,
            ContinuousAssociationType.SPEARMAN,
            (2, 2),
        ),
    ],
)
def test_compute_df_correlations(
    df: pd.DataFrame,
    cat_features: List[str],
    cat_corr: CategoricalAssociationType,
    cont_corr: ContinuousAssociationType,
    expected_shape: Tuple[int, int],
):
    result = CorrelationCalculator.compute_df_correlations(
        categorical_correlation_type=cat_corr,
        continuous_correlation_type=cont_corr,
        dataset=df,
        cat_features=cat_features,
    )

    assert result.shape == expected_shape
    assert (result.values >= 0).all() and (result.values <= 1).all()


@pytest.mark.unit
@pytest.mark.parametrize(
    "df_real, df_synthetic, cat_features, cat_corr, cont_corr, expected_shape",
    [
        (
            pd.DataFrame({"x": [1, 2, 3]}),
            pd.DataFrame({"x": [3, 2, 1]}),
            [],
            CategoricalAssociationType.CRAMERS_V,
            ContinuousAssociationType.PEARSON,
            (1, 1),
        ),
        (
            pd.DataFrame({"cat": ["a", "b", "a"], "num": [1, 2, 3]}),
            pd.DataFrame({"cat": ["b", "a", "b"], "num": [3, 2, 1]}),
            ["cat"],
            CategoricalAssociationType.THEILS_U,
            ContinuousAssociationType.SPEARMAN,
            (2, 2),
        ),
    ],
)
def test_compute_df_correlation_difference(
    df_real: pd.DataFrame,
    df_synthetic: pd.DataFrame,
    cat_features: List[str],
    cat_corr: CategoricalAssociationType,
    cont_corr: ContinuousAssociationType,
    expected_shape: Tuple[int, int],
):
    result = CorrelationCalculator.compute_df_correlation_difference(
        categorical_correlation_type=cat_corr,
        continuous_correlation_type=cont_corr,
        dataset_real=df_real,
        dataset_synthetic=df_synthetic,
        cat_features=cat_features,
    )

    assert result.shape == expected_shape
    assert (result.values >= 0).all() and (result.values <= 1).all()


@pytest.mark.unit
@pytest.mark.parametrize(
    "df_real, df_synthetic, cat_features, cat_corr, cont_corr, "
    "distance_metric, expected_distance",
    [
        (
            pd.DataFrame({"x": [1, 2, 3]}),
            pd.DataFrame({"x": [1, 2, 3]}),
            [],
            CategoricalAssociationType.CRAMERS_V,
            ContinuousAssociationType.PEARSON,
            VectorDistanceMetric.L2,
            0.0,
        ),
        (
            pd.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3]}),
            pd.DataFrame({"x": [3, 2, 1], "y": [88, -11, 2]}),
            [],
            CategoricalAssociationType.CRAMERS_V,
            ContinuousAssociationType.PEARSON,
            VectorDistanceMetric.L2,
            None,
        ),
    ],
)
def test_compute_correlation_distance(
    mocker: MockerFixture,
    df_real: pd.DataFrame,
    df_synthetic: pd.DataFrame,
    cat_features: List[str],
    cat_corr: CategoricalAssociationType,
    cont_corr: ContinuousAssociationType,
    distance_metric: VectorDistanceMetric,
    expected_distance: Optional[float],
):
    spy = mocker.spy(obj=VectorDistanceCalculator, name="compute")

    result = CorrelationCalculator.compute_correlation_distance(
        categorical_correlation_type=cat_corr,
        continuous_correlation_type=cont_corr,
        distance_metric=distance_metric,
        dataset_real=df_real,
        dataset_synthetic=df_synthetic,
        cat_features=cat_features,
    )

    spy.assert_called_once()
    assert spy.call_args.kwargs["metric"] == distance_metric

    if expected_distance is not None:
        assert isclose(result, expected_distance, rel_tol=1e-7)
    else:
        assert result > 0
