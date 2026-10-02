from collections.abc import Callable
from typing import Any

import numpy as np

from .result import SimulationResult


def simulate(
    model: Callable[[np.random.Generator], Any],
    n: int = 10_000,
    seed: int | None = None,
) -> SimulationResult:
    """
    Run a Monte Carlo simulation.

    Parameters
    ----------
    model:
        Function that receives a NumPy random generator and
        returns one numeric simulation result.

    n:
        Number of simulation runs.

    seed:
        Optional random seed for reproducibility.

    Returns
    -------
    SimulationResult
        Results of the simulation.
    """

    if not isinstance(n, (int, np.integer)):
        raise TypeError("n must be an integer.")

    if n <= 0:
        raise ValueError("n must be greater than 0.")

    if not callable(model):
        raise TypeError("model must be callable.")

    if seed is not None and not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer or None.")

    rng = np.random.default_rng(seed)

    values = []

    for _ in range(n):
        value = model(rng)

        if not np.isscalar(value):
            raise ValueError(
                "The model must return exactly one scalar value "
                "per simulation run."
            )

        if not isinstance(
            value,
            (int, float, np.integer, np.floating),
        ):
            raise TypeError(
                "The model must return a numeric value."
            )

        if not np.isfinite(value):
            raise ValueError(
                "The model returned a non-finite value."
            )

        values.append(value)

    values = np.asarray(values, dtype=float)

    return SimulationResult(
        values=values,
        n=n,
        seed=seed,
    )