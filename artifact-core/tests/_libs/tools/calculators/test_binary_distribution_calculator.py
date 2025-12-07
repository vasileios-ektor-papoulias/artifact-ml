from typing import Dict, Hashable

import numpy as np
import pytest
from artifact_core._libs.tools.calculators.binary_distribution_calculator import (
    BinaryDistributionCalculator,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    "prob_pos, pos_idx, expected",
    [
        (0.7, 0, np.array([0.7, 0.3])),
        (0.7, 1, np.array([0.3, 0.7])),
        (0.5, 0, np.array([0.5, 0.5])),
        (1.0, 0, np.array([1.0, 0.0])),
        (0.0, 0, np.array([0.0, 1.0])),
    ],
)
def test_compute_probs(prob_pos: float, pos_idx: int, expected: np.ndarray):
    result = BinaryDistributionCalculator.compute_probs(prob_pos=prob_pos, pos_idx=pos_idx)
    np.testing.assert_array_almost_equal(result, expected, decimal=5)


@pytest.mark.unit
@pytest.mark.parametrize(
    "prob_pos, pos_idx",
    [
        (0.8, 0),
        (0.8, 1),
        (0.5, 0),
    ],
)
def test_compute_probs_sums_to_one(prob_pos: float, pos_idx: int):
    result = BinaryDistributionCalculator.compute_probs(prob_pos=prob_pos, pos_idx=pos_idx)
    assert result.sum() == pytest.approx(expected=1.0)


@pytest.mark.unit
@pytest.mark.parametrize(
    "id_to_prob_pos, pos_idx",
    [
        ({0: 0.7, 1: 0.3, 2: 0.5}, 0),
        ({0: 0.7, 1: 0.3, 2: 0.5}, 1),
        ({"a": 0.9, "b": 0.1}, 0),
    ],
)
def test_compute_id_to_probs(id_to_prob_pos: Dict[Hashable, float], pos_idx: int):
    result = BinaryDistributionCalculator.compute_id_to_probs(
        id_to_prob_pos=id_to_prob_pos, pos_idx=pos_idx
    )
    assert result.keys() == id_to_prob_pos.keys()
    for identifier in result:
        assert result[identifier].shape == (2,)
        assert result[identifier].sum() == pytest.approx(expected=1.0)


@pytest.mark.unit
@pytest.mark.parametrize(
    "prob, expected_complement",
    [
        (0.7, 0.3),
        (0.5, 0.5),
        (0.0, 1.0),
        (1.0, 0.0),
    ],
)
def test_compute_prob_complement(prob: float, expected_complement: float):
    result = BinaryDistributionCalculator.compute_prob_complement(prob=prob, eps=1e-15)
    assert result == pytest.approx(expected=expected_complement, abs=1e-5)


@pytest.mark.unit
@pytest.mark.parametrize(
    "id_to_prob",
    [
        {0: 0.7, 1: 0.3, 2: 0.5},
        {"x": 0.9, "y": 0.1},
    ],
)
def test_compute_id_to_prob_complement(id_to_prob: Dict[Hashable, float]):
    result = BinaryDistributionCalculator.compute_id_to_prob_complement(id_to_prob=id_to_prob)
    assert result.keys() == id_to_prob.keys()
    for identifier in result:
        assert result[identifier] == pytest.approx(expected=1.0 - id_to_prob[identifier], abs=1e-5)


@pytest.mark.unit
@pytest.mark.parametrize(
    "prob_pos, pos_idx",
    [
        (0.0, 0),
        (1.0, 0),
        (0.0, 1),
        (1.0, 1),
    ],
)
def test_extreme_prob_values_clipped(prob_pos: float, pos_idx: int):
    result = BinaryDistributionCalculator.compute_probs(prob_pos=prob_pos, pos_idx=pos_idx)
    assert not np.any(np.isnan(result))
    assert result.sum() == pytest.approx(expected=1.0, abs=1e-10)
