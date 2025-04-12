from math import isclose
from typing import List

import pandas as pd
import pytest
from artifact_core.libs.implementation.pairwsie_correlation.calculator import (
    CategoricalAssociationType,
    ContinuousAssociationType,
    PairwiseCorrelationCalculator,
)
from artifact_core.libs.utils.vector_distance_calculator import (
    VectorDistanceMetric,
)


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
def test_compute_df_correlations(df, cat_features, cat_corr, cont_corr, expected_shape):
    result = PairwiseCorrelationCalculator.compute_df_correlations(
        categorical_correlation_type=cat_corr,
        continuous_correlation_type=cont_corr,
        dataset=df,
        ls_cat_features=cat_features,
    )

    assert result.shape == expected_shape
    assert (result.values >= 0).all() and (result.values <= 1).all(), (
        "Correlation values must be between 0 and 1"
    )


@pytest.mark.parametrize(
    "df_real, df_synth, cat_features, cat_corr, cont_corr, expected_shape",
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
    df_real, df_synth, cat_features, cat_corr, cont_corr, expected_shape
):
    result = PairwiseCorrelationCalculator.compute_df_correlation_difference(
        categorical_correlation_type=cat_corr,
        continuous_correlation_type=cont_corr,
        dataset_real=df_real,
        dataset_synthetic=df_synth,
        ls_cat_features=cat_features,
    )

    assert result.shape == expected_shape
    assert (result.values >= 0).all() and (result.values <= 1).all(), (
        "Difference values must be between 0 and 1"
    )


@pytest.mark.parametrize(
    "df_real, df_synth, cat_features, cat_corr, cont_corr, distance_metric, expected_distance",
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
    df_real: pd.DataFrame,
    df_synth: pd.DataFrame,
    cat_features: List[str],
    cat_corr: CategoricalAssociationType,
    cont_corr: ContinuousAssociationType,
    distance_metric: VectorDistanceMetric,
    expected_distance: float,
):
    result = PairwiseCorrelationCalculator.compute_correlation_distance(
        categorical_correlation_type=cat_corr,
        continuous_correlation_type=cont_corr,
        distance_metric=distance_metric,
        dataset_real=df_real,
        dataset_synthetic=df_synth,
        ls_cat_features=cat_features,
    )
    if expected_distance is not None:
        assert isclose(result, expected_distance, rel_tol=1e-7), (
            f"Expected distance {expected_distance}, got {result}"
        )
    else:
        assert result > 0, f"Expected positive distance, got {result}"
