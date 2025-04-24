from enum import Enum
from math import isclose, sqrt
from typing import cast

import numpy as np
import pytest
from artifact_core.libs.utils.vector_distance_calculator import (
    VectorDistanceCalculator,
    VectorDistanceMetric,
)


class FakeMetric(Enum):
    UNSUPPORTED = "unsupported"


def test_unsupported_metric():
    with pytest.raises(
        ValueError,
        match=f"Unsupported distance metric: {FakeMetric.UNSUPPORTED}",
    ):
        VectorDistanceCalculator.compute(
            cast(VectorDistanceMetric, FakeMetric.UNSUPPORTED),
            np.array([1, 2]),
            np.array([1, 2]),
        )


@pytest.mark.parametrize(
    "metric, v1, v2, expected",
    [
        (VectorDistanceMetric.L2, np.array([0, 0]), np.array([3, 4]), 5.0),
        (
            VectorDistanceMetric.MAE,
            np.array([1, 2, 3]),
            np.array([2, 2, 5]),
            1.0,
        ),
        (
            VectorDistanceMetric.RMSE,
            np.array([1, 2, 3]),
            np.array([2, 2, 5]),
            sqrt(5 / 3),
        ),
        (
            VectorDistanceMetric.COSINE_SIMILARITY,
            np.array([1, 0]),
            np.array([0, 1]),
            0.0,
        ),
        (
            VectorDistanceMetric.COSINE_SIMILARITY,
            np.array([0, 0]),
            np.array([1, 2]),
            np.nan,
        ),
    ],
)
def test_vector_distance(
    metric: VectorDistanceMetric,
    v1: np.ndarray,
    v2: np.ndarray,
    expected: float,
):
    result = VectorDistanceCalculator.compute(metric, v1, v2)
    if np.isnan(expected):
        assert np.isnan(result), f"Expected NaN for {metric}, got {result}"
    else:
        assert isclose(result, expected, rel_tol=1e-5), (
            f"For {metric}, expected {expected}, got {result}"
        )
