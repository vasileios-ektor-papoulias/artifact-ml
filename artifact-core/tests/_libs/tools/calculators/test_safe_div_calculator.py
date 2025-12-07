import pytest
from artifact_core._libs.tools.calculators.safe_div_calculator import SafeDivCalculator


@pytest.mark.unit
@pytest.mark.parametrize(
    "num, denom, expected",
    [
        (10.0, 2.0, 5.0),
        (1.0, 4.0, 0.25),
        (0.0, 5.0, 0.0),
        (-10.0, 2.0, -5.0),
        (10.0, -2.0, -5.0),
        (10.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (-5.0, 0.0, 0.0),
    ],
)
def test_compute(num: float, denom: float, expected: float):
    result = SafeDivCalculator.compute(num=num, denom=denom)
    assert result == pytest.approx(expected=expected)

