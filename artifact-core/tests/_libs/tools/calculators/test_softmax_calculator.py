from typing import Dict, Hashable

import numpy as np
import pytest
from artifact_core._libs.tools.calculators.softmax_calculator import SoftmaxCalculator


@pytest.mark.unit
@pytest.mark.parametrize(
    "logits, expected_sum",
    [
        (np.array([1.0, 2.0, 3.0]), 1.0),
        (np.array([0.0, 0.0, 0.0]), 1.0),
        (np.array([-1.0, 0.0, 1.0]), 1.0),
        (np.array([100.0, 0.0]), 1.0),
    ],
)
def test_compute_probs_sums_to_one(logits: np.ndarray, expected_sum: float):
    probs = SoftmaxCalculator.compute_probs(logits=logits)
    assert probs.sum() == pytest.approx(expected=expected_sum)


@pytest.mark.unit
@pytest.mark.parametrize(
    "logits, expected_argmax",
    [
        (np.array([1.0, 2.0, 3.0]), 2),
        (np.array([3.0, 2.0, 1.0]), 0),
        (np.array([0.0, 5.0, 0.0]), 1),
    ],
)
def test_compute_probs_argmax(logits: np.ndarray, expected_argmax: int):
    probs = SoftmaxCalculator.compute_probs(logits=logits)
    assert np.argmax(probs) == expected_argmax


@pytest.mark.unit
def test_compute_probs_empty():
    logits = np.array([])
    probs = SoftmaxCalculator.compute_probs(logits=logits)
    assert probs.size == 0


@pytest.mark.unit
@pytest.mark.parametrize(
    "logits",
    [
        (np.array([0.0, 0.0, 0.0])),
        (np.array([1.0, 2.0, 3.0])),
        (np.array([-1.0, 0.0, 1.0])),
    ],
)
def test_compute_probs_all_positive(logits: np.ndarray):
    probs = SoftmaxCalculator.compute_probs(logits=logits)
    assert np.all(probs > 0)


@pytest.mark.unit
@pytest.mark.parametrize(
    "id_to_logits",
    [
        {0: np.array([1.0, 2.0]), 1: np.array([0.0, 0.0])},
        {"a": np.array([1.0, 1.0, 1.0])},
    ],
)
def test_compute_probs_multiple(id_to_logits: Dict[Hashable, np.ndarray]):
    result = SoftmaxCalculator.compute_probs_multiple(id_to_logits=id_to_logits)
    assert result.keys() == id_to_logits.keys()
    for identifier in result:
        assert result[identifier].sum() == pytest.approx(expected=1.0)


@pytest.mark.unit
@pytest.mark.parametrize(
    "probs, expected_shape",
    [
        (np.array([0.5, 0.5]), (2,)),
        (np.array([0.1, 0.2, 0.7]), (3,)),
        (np.array([1.0, 0.0, 0.0]), (3,)),
    ],
)
def test_compute_logits_shape(probs: np.ndarray, expected_shape: tuple):
    logits = SoftmaxCalculator.compute_logits(probs=probs)
    assert logits.shape == expected_shape


@pytest.mark.unit
def test_compute_logits_zero_prob_gives_neg_inf():
    probs = np.array([1.0, 0.0, 0.0])
    logits = SoftmaxCalculator.compute_logits(probs=probs)
    assert logits[0] > logits[1]
    assert logits[1] == -np.inf
    assert logits[2] == -np.inf


@pytest.mark.unit
def test_compute_logits_empty():
    probs = np.array([])
    logits = SoftmaxCalculator.compute_logits(probs=probs)
    assert logits.size == 0


@pytest.mark.unit
@pytest.mark.parametrize(
    "id_to_probs",
    [
        {0: np.array([0.5, 0.5]), 1: np.array([0.3, 0.7])},
        {"x": np.array([0.1, 0.2, 0.7])},
    ],
)
def test_compute_logits_multiple(id_to_probs: Dict[Hashable, np.ndarray]):
    result = SoftmaxCalculator.compute_logits_multiple(id_to_probs=id_to_probs)
    assert result.keys() == id_to_probs.keys()
    for identifier in result:
        assert result[identifier].shape == id_to_probs[identifier].shape
