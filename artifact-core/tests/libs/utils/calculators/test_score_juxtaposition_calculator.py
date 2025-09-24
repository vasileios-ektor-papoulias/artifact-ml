from typing import Dict, List, Optional

import numpy as np
import pytest
from artifact_core.libs.utils.calculators.score_juxtaposition_calculator import (
    ScoreJuxtapositionCalculator,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    "dict_scores_real, dict_scores_synthetic, ls_keys, expected",
    [
        (
            {"a": 1.0, "b": 2.0},
            {"a": 1.5, "b": 2.5},
            None,
            {"a": np.array([1.0, 1.5]), "b": np.array([2.0, 2.5])},
        ),
        (
            {"x": 10.0, "y": 20.0, "z": 30.0},
            {"x": 15.0, "y": 25.0, "z": 35.0},
            ["y", "z"],
            {"y": np.array([20.0, 25.0]), "z": np.array([30.0, 35.0])},
        ),
        ({}, {}, None, {}),
    ],
)
def test_juxtaposition_array(
    dict_scores_real: Dict[str, float],
    dict_scores_synthetic: Dict[str, float],
    ls_keys: Optional[List[str]],
    expected: Dict[str, np.ndarray],
):
    result = ScoreJuxtapositionCalculator.juxtapose_score_collections(
        dict_scores_real=dict_scores_real,
        dict_scores_synthetic=dict_scores_synthetic,
        ls_keys=ls_keys,
    )
    assert result.keys() == expected.keys()
    for key in result:
        assert np.allclose(result[key], expected[key]), f"Mismatch for key '{key}'"


@pytest.mark.unit
@pytest.mark.parametrize(
    "score_real, score_synthetic, expected",
    [
        (0.0, 1.0, np.array([0.0, 1.0])),
        (2.5, 2.5, np.array([2.5, 2.5])),
        (-1.0, 1.0, np.array([-1.0, 1.0])),
    ],
)
def test_juxtapose_scores(score_real: float, score_synthetic: float, expected: np.ndarray):
    result = ScoreJuxtapositionCalculator.juxtapose_scores(score_real, score_synthetic)
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"
