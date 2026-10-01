import numpy as np
import pytest

from simplemc import SimulationResult


def test_result_statistics():
    result = SimulationResult(
        values=np.array([1, 2, 3, 4, 5]),
        n=5,
    )

    assert result.mean == 3.0
    assert result.min == 1.0
    assert result.max == 5.0
    assert result.quantile(0.5) == 3.0


def test_invalid_quantile():
    result = SimulationResult(
        values=np.array([1, 2, 3]),
        n=3,
    )

    with pytest.raises(ValueError):
        result.quantile(1.5)