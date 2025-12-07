from typing import Dict, Hashable

import pytest
from artifact_core._libs.tools.calculators.sigmoid_calculator import SigmoidCalculator


@pytest.mark.unit
@pytest.mark.parametrize(
    "logit, expected_prob",
    [
        (0.0, 0.5),
        (100.0, 1.0),
        (-100.0, 0.0),
    ],
)
def test_compute_prob(logit: float, expected_prob: float):
    result = SigmoidCalculator.compute_prob(logit=logit)
    assert result == pytest.approx(expected=expected_prob, abs=1e-5)


@pytest.mark.unit
@pytest.mark.parametrize(
    "logit",
    [0.0, 1.0, -1.0, 5.0, -5.0],
)
def test_compute_prob_in_range(logit: float):
    result = SigmoidCalculator.compute_prob(logit=logit)
    assert 0.0 <= result <= 1.0


@pytest.mark.unit
@pytest.mark.parametrize(
    "id_to_logit",
    [
        {0: 0.0, 1: 1.0, 2: -1.0},
        {"a": 0.0, "b": 100.0},
    ],
)
def test_compute_probs_multiple(id_to_logit: Dict[Hashable, float]):
    result = SigmoidCalculator.compute_probs_multiple(id_to_logit=id_to_logit)
    assert result.keys() == id_to_logit.keys()
    for identifier in result:
        assert 0.0 <= result[identifier] <= 1.0


@pytest.mark.unit
@pytest.mark.parametrize(
    "prob, expected_logit",
    [
        (0.5, 0.0),
    ],
)
def test_compute_logit(prob: float, expected_logit: float):
    result = SigmoidCalculator.compute_logit(prob=prob)
    assert result == pytest.approx(expected=expected_logit, abs=1e-5)


@pytest.mark.unit
@pytest.mark.parametrize(
    "prob",
    [0.5, 0.1, 0.9, 0.99, 0.01],
)
def test_compute_logit_roundtrip(prob: float):
    logit = SigmoidCalculator.compute_logit(prob=prob)
    recovered_prob = SigmoidCalculator.compute_prob(logit=logit)
    assert recovered_prob == pytest.approx(expected=prob, abs=1e-5)


@pytest.mark.unit
@pytest.mark.parametrize(
    "prob",
    [0.0, 1.0],
)
def test_compute_logit_extreme_values_clipped(prob: float):
    logit = SigmoidCalculator.compute_logit(prob=prob)
    assert isinstance(logit, float)


@pytest.mark.unit
@pytest.mark.parametrize(
    "id_to_prob",
    [
        {0: 0.5, 1: 0.8, 2: 0.2},
        {"x": 0.1, "y": 0.9},
    ],
)
def test_compute_logits_multiple(id_to_prob: Dict[Hashable, float]):
    result = SigmoidCalculator.compute_logits_multiple(id_to_prob=id_to_prob)
    assert result.keys() == id_to_prob.keys()
    for identifier in result:
        assert isinstance(result[identifier], float)
