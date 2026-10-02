import numpy as np
import pytest

from simplemc import simulate


def test_simulation_returns_result():
    result = simulate(
        lambda rng: rng.normal(),
        n=100,
        seed=42,
    )

    assert result.n == 100
    assert len(result.values) == 100


def test_seed_reproducibility():
    result1 = simulate(
        lambda rng: rng.normal(),
        n=1000,
        seed=42,
    )

    result2 = simulate(
        lambda rng: rng.normal(),
        n=1000,
        seed=42,
    )

    assert np.array_equal(result1.values, result2.values)


def test_different_seeds():
    result1 = simulate(
        lambda rng: rng.normal(),
        n=1000,
        seed=42,
    )

    result2 = simulate(
        lambda rng: rng.normal(),
        n=1000,
        seed=123,
    )

    assert not np.array_equal(result1.values, result2.values)


def test_invalid_n_type():
    with pytest.raises(TypeError):
        simulate(lambda rng: 1, n=10.5)


def test_invalid_n_value():
    with pytest.raises(ValueError):
        simulate(lambda rng: 1, n=0)


def test_invalid_model():
    with pytest.raises(TypeError):
        simulate("not a function")


def test_invalid_seed():
    with pytest.raises(TypeError):
        simulate(lambda rng: 1, seed="42")


def test_model_must_return_scalar():
    with pytest.raises(ValueError):
        simulate(
            lambda rng: np.array([1, 2, 3]),
            n=10,
        )


def test_model_must_return_numeric():
    with pytest.raises(TypeError):
        simulate(
            lambda rng: "hello",
            n=10,
        )


def test_model_must_return_finite_value():
    with pytest.raises(ValueError):
        simulate(
            lambda rng: np.inf,
            n=10,
        )