from dataclasses import dataclass

import numpy as np


@dataclass
class SimulationResult:
    """Results returned by a Monte Carlo simulation."""

    values: np.ndarray
    n: int
    seed: int | None = None

    def __post_init__(self) -> None:
        """Validate the simulation result."""

        if not isinstance(self.values, np.ndarray):
            raise TypeError("values must be a NumPy array.")

        if self.values.ndim != 1:
            raise ValueError("values must be a one-dimensional array.")

        if not isinstance(self.n, (int, np.integer)):
            raise TypeError("n must be an integer.")

        if self.n <= 0:
            raise ValueError("n must be greater than 0.")

        if len(self.values) != self.n:
            raise ValueError(
                "n must match the number of simulated values."
            )

        if not np.issubdtype(self.values.dtype, np.number):
            raise TypeError("values must contain numeric data.")

        if not np.all(np.isfinite(self.values)):
            raise ValueError(
                "values must contain only finite values."
            )

        if self.seed is not None:
            if not isinstance(self.seed, (int, np.integer)):
                raise TypeError("seed must be an integer or None.")

    @property
    def mean(self) -> float:
        """Return the sample mean."""
        return float(np.mean(self.values))

    @property
    def std(self) -> float:
        """Return the sample standard deviation."""

        if self.n < 2:
            raise ValueError(
                "At least two simulation results are required "
                "to calculate the sample standard deviation."
            )

        return float(np.std(self.values, ddof=1))

    @property
    def min(self) -> float:
        """Return the minimum simulated value."""
        return float(np.min(self.values))

    @property
    def max(self) -> float:
        """Return the maximum simulated value."""
        return float(np.max(self.values))

    def quantile(self, q: float) -> float:
        """Return a simulated quantile."""

        if not isinstance(q, (int, float, np.integer, np.floating)):
            raise TypeError("q must be a number.")

        if not np.isfinite(q):
            raise ValueError("q must be finite.")

        if not 0 <= q <= 1:
            raise ValueError("q must be between 0 and 1.")

        return float(np.quantile(self.values, q))

    def __repr__(self) -> str:
        return (
            f"SimulationResult("
            f"n={self.n}, "
            f"mean={self.mean:.6f}, "
            f"std={self.std:.6f}, "
            f"min={self.min:.6f}, "
            f"max={self.max:.6f})"
        )