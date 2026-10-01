from dataclasses import dataclass

import numpy as np


@dataclass
class SimulationResult:
    """Results returned by a Monte Carlo simulation."""

    values: np.ndarray
    n: int
    seed: int | None = None

    @property
    def mean(self) -> float:
        """Return the sample mean."""
        return float(np.mean(self.values))

    @property
    def std(self) -> float:
        """Return the sample standard deviation."""
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