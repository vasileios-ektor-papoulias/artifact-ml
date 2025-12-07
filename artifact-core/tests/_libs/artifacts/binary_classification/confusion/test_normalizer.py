import numpy as np
import pytest
from artifact_core._libs.artifacts.binary_classification.confusion.normalizer import (
    ConfusionMatrixNormalizationStrategy,
    ConfusionMatrixNormalizer,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    "arr_cm, normalization, expected",
    [
        (
            np.array([[4, 2], [1, 3]]),
            ConfusionMatrixNormalizationStrategy.NONE,
            np.array([[4.0, 2.0], [1.0, 3.0]]),
        ),
        (
            np.array([[4, 2], [1, 3]]),
            ConfusionMatrixNormalizationStrategy.TRUE,
            np.array([[4 / 6, 2 / 6], [1 / 4, 3 / 4]]),
        ),
        (
            np.array([[4, 2], [1, 3]]),
            ConfusionMatrixNormalizationStrategy.PRED,
            np.array([[4 / 5, 2 / 5], [1 / 5, 3 / 5]]),
        ),
        (
            np.array([[4, 2], [1, 3]]),
            ConfusionMatrixNormalizationStrategy.ALL,
            np.array([[4 / 10, 2 / 10], [1 / 10, 3 / 10]]),
        ),
    ],
)
def test_normalize_cm(
    arr_cm: np.ndarray,
    normalization: ConfusionMatrixNormalizationStrategy,
    expected: np.ndarray,
):
    result = ConfusionMatrixNormalizer.normalize_cm(arr_cm=arr_cm, normalization=normalization)
    np.testing.assert_array_almost_equal(result, expected)


@pytest.mark.unit
def test_normalize_cm_preserves_original():
    arr_cm = np.array([[4, 2], [1, 3]])
    original = arr_cm.copy()
    ConfusionMatrixNormalizer.normalize_cm(
        arr_cm=arr_cm, normalization=ConfusionMatrixNormalizationStrategy.ALL
    )
    np.testing.assert_array_equal(arr_cm, original)


@pytest.mark.unit
@pytest.mark.parametrize(
    "normalization",
    [
        ConfusionMatrixNormalizationStrategy.TRUE,
        ConfusionMatrixNormalizationStrategy.PRED,
        ConfusionMatrixNormalizationStrategy.ALL,
    ],
)
def test_normalize_cm_zero_handling(normalization: ConfusionMatrixNormalizationStrategy):
    arr_cm = np.array([[0, 0], [0, 0]])
    result = ConfusionMatrixNormalizer.normalize_cm(arr_cm=arr_cm, normalization=normalization)
    assert not np.any(np.isnan(result))
    assert not np.any(np.isinf(result))
