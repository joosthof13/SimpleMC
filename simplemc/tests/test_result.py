import numpy as np
import pytest

from simplemc.result import SimulationResult


def test_mean():
    result = SimulationResult(
        values=np.array([1, 2, 3, 4, 5]),
        n=5,
    )

    assert result.mean == 3.0


def test_std():
    result = SimulationResult(
        values=np.array([1, 2, 3, 4, 5]),
        n=5,
    )

    assert result.std == pytest.approx(np.std(
        [1, 2, 3, 4, 5],
        ddof=1,
    ))


def test_min():
    result = SimulationResult(
        values=np.array([1, 2, 3]),
        n=3,
    )

    assert result.min == 1


def test_max():
    result = SimulationResult(
        values=np.array([1, 2, 3]),
        n=3,
    )

    assert result.max == 3


def test_quantile():
    result = SimulationResult(
        values=np.arange(100),
        n=100,
    )

    assert result.quantile(0.5) == pytest.approx(49.5)


def test_invalid_quantile():
    result = SimulationResult(
        values=np.array([1, 2, 3]),
        n=3,
    )

    with pytest.raises(ValueError):
        result.quantile(1.5)


def test_mismatched_n():
    with pytest.raises(ValueError):
        SimulationResult(
            values=np.array([1, 2, 3]),
            n=10,
        )