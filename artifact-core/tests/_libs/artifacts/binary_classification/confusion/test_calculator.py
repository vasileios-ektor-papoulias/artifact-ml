from typing import Dict, Hashable, List

import numpy as np
import pytest
from artifact_core._libs.artifacts.binary_classification.confusion.calculator import (
    NormalizedConfusionCalculator,
)
from artifact_core._libs.artifacts.binary_classification.confusion.normalizer import (
    ConfusionMatrixNormalizationStrategy,
    ConfusionMatrixNormalizer,
)
from artifact_core._libs.artifacts.binary_classification.confusion.raw import (
    ConfusionMatrixCell,
    RawConfusionCalculator,
)
from pytest_mock import MockerFixture


@pytest.mark.unit
@pytest.mark.parametrize(
    "true, predicted, normalization, expected_matrix",
    [
        (
            {0: True, 1: False},
            {0: True, 1: False},
            ConfusionMatrixNormalizationStrategy.NONE,
            [[1.0, 0.0], [0.0, 1.0]],
        ),
        (
            {0: True, 1: False},
            {0: True, 1: False},
            ConfusionMatrixNormalizationStrategy.TRUE,
            [[1.0, 0.0], [0.0, 1.0]],
        ),
        (
            {0: True, 1: False},
            {0: True, 1: False},
            ConfusionMatrixNormalizationStrategy.ALL,
            [[0.5, 0.0], [0.0, 0.5]],
        ),
        (
            {0: True, 1: True, 2: False, 3: False},
            {0: True, 1: False, 2: True, 3: False},
            ConfusionMatrixNormalizationStrategy.ALL,
            [[0.25, 0.25], [0.25, 0.25]],
        ),
    ],
)
def test_compute_normalized_confusion_matrix(
    mocker: MockerFixture,
    true: Dict[Hashable, bool],
    predicted: Dict[Hashable, bool],
    normalization: ConfusionMatrixNormalizationStrategy,
    expected_matrix: List[List[float]],
):
    spy_raw = mocker.spy(obj=RawConfusionCalculator, name="_compute_confusion_matrix")
    spy_normalizer = mocker.spy(obj=ConfusionMatrixNormalizer, name="normalize_cm")
    result = NormalizedConfusionCalculator.compute_normalized_confusion_matrix(
        true=true, predicted=predicted, normalization=normalization
    )
    np.testing.assert_array_almost_equal(result, expected_matrix)
    spy_raw.assert_called_once()
    spy_normalizer.assert_called_once()
    assert spy_normalizer.call_args.kwargs["normalization"] == normalization


@pytest.mark.unit
@pytest.mark.parametrize(
    "true, predicted, normalization_types, expected_count",
    [
        (
            {0: True, 1: False},
            {0: True, 1: False},
            [ConfusionMatrixNormalizationStrategy.TRUE],
            1,
        ),
        (
            {0: True, 1: False, 2: True},
            {0: True, 1: False, 2: False},
            [ConfusionMatrixNormalizationStrategy.TRUE, ConfusionMatrixNormalizationStrategy.PRED],
            2,
        ),
        (
            {0: True, 1: False},
            {0: True, 1: False},
            list(ConfusionMatrixNormalizationStrategy),
            4,
        ),
    ],
)
def test_compute_confusion_matrix_multiple_normalizations(
    mocker: MockerFixture,
    true: Dict[Hashable, bool],
    predicted: Dict[Hashable, bool],
    normalization_types: List[ConfusionMatrixNormalizationStrategy],
    expected_count: int,
):
    spy_normalizer = mocker.spy(obj=ConfusionMatrixNormalizer, name="normalize_cm")
    result = NormalizedConfusionCalculator.compute_confusion_matrix_multiple_normalizations(
        true=true, predicted=predicted, normalization_types=normalization_types
    )
    assert len(result) == expected_count
    assert set(result.keys()) == set(normalization_types)
    assert spy_normalizer.call_count == expected_count


@pytest.mark.unit
@pytest.mark.parametrize(
    "true, predicted, cell, normalization, expected",
    [
        (
            {0: True, 1: False},
            {0: True, 1: False},
            ConfusionMatrixCell.TRUE_POSITIVE,
            ConfusionMatrixNormalizationStrategy.TRUE,
            1.0,
        ),
        (
            {0: True, 1: False},
            {0: True, 1: False},
            ConfusionMatrixCell.TRUE_NEGATIVE,
            ConfusionMatrixNormalizationStrategy.TRUE,
            1.0,
        ),
        (
            {0: True, 1: False},
            {0: True, 1: False},
            ConfusionMatrixCell.TRUE_POSITIVE,
            ConfusionMatrixNormalizationStrategy.ALL,
            0.5,
        ),
        (
            {0: True, 1: True, 2: False, 3: False},
            {0: True, 1: False, 2: True, 3: False},
            ConfusionMatrixCell.FALSE_POSITIVE,
            ConfusionMatrixNormalizationStrategy.ALL,
            0.25,
        ),
    ],
)
def test_compute_normalized_confusion_count(
    mocker: MockerFixture,
    true: Dict[Hashable, bool],
    predicted: Dict[Hashable, bool],
    cell: ConfusionMatrixCell,
    normalization: ConfusionMatrixNormalizationStrategy,
    expected: float,
):
    spy_normalizer = mocker.spy(obj=ConfusionMatrixNormalizer, name="normalize_cm")
    result = NormalizedConfusionCalculator.compute_normalized_confusion_count(
        confusion_matrix_cell=cell, true=true, predicted=predicted, normalization=normalization
    )
    assert result == pytest.approx(expected=expected)
    spy_normalizer.assert_called_once()


@pytest.mark.unit
@pytest.mark.parametrize(
    "true, predicted, normalization, expected_total",
    [
        (
            {0: True, 1: False},
            {0: True, 1: False},
            ConfusionMatrixNormalizationStrategy.ALL,
            1.0,
        ),
        (
            {0: True, 1: True, 2: False, 3: False},
            {0: True, 1: False, 2: True, 3: False},
            ConfusionMatrixNormalizationStrategy.ALL,
            1.0,
        ),
    ],
)
def test_compute_dict_normalized_confusion_counts(
    mocker: MockerFixture,
    true: Dict[Hashable, bool],
    predicted: Dict[Hashable, bool],
    normalization: ConfusionMatrixNormalizationStrategy,
    expected_total: float,
):
    spy_normalizer = mocker.spy(obj=ConfusionMatrixNormalizer, name="normalize_cm")
    result = NormalizedConfusionCalculator.compute_dict_normalized_confusion_counts(
        true=true, predicted=predicted, normalization=normalization
    )
    assert set(result.keys()) == set(ConfusionMatrixCell)
    assert sum(result.values()) == pytest.approx(expected=expected_total)
    spy_normalizer.assert_called_once()
