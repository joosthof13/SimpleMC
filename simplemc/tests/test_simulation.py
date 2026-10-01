import numpy as np
import pytest

from simplemc import SimulationResult, simulate


def test_simulation_returns_result():
    result = simulate(
        lambda rng: rng.normal(),
        n=100,
        seed=42,
    )

    assert isinstance(result, SimulationResult)
    assert result.n == 100
    assert len(result.values) == 100


def test_seed_is_reproducible():
    result_1 = simulate(
        lambda rng: rng.normal(),
        n=100,
        seed=42,
    )

    result_2 = simulate(
        lambda rng: rng.normal(),
        n=100,
        seed=42,
    )

    np.testing.assert_array_equal(
        result_1.values,
        result_2.values,
    )


def test_different_seeds_produce_different_results():
    result_1 = simulate(
        lambda rng: rng.normal(),
        n=100,
        seed=42,
    )

    result_2 = simulate(
        lambda rng: rng.normal(),
        n=100,
        seed=43,
    )

    assert not np.array_equal(
        result_1.values,
        result_2.values,
    )


def test_invalid_n():
    with pytest.raises(ValueError):
        simulate(lambda rng: 1, n=0)


def test_invalid_model():
    with pytest.raises(TypeError):
        simulate("not a function", n=10)