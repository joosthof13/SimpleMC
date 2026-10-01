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
        returns one simulation result.

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

    rng = np.random.default_rng(seed)

    values = [model(rng) for _ in range(n)]
    values = np.asarray(values)

    if values.ndim != 1:
        raise ValueError(
            "The model must return exactly one value per simulation run."
        )

    return SimulationResult(
        values=values,
        n=n,
        seed=seed,
    )