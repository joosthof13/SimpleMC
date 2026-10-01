# SimpleMC

SimpleMC is a lightweight Python package for running Monte Carlo simulations.

The goal of SimpleMC is to provide a simple interface for defining a stochastic model, running it repeatedly, and analyzing the resulting samples.

## Installation

### Development installation

Clone the repository and activate the virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install SimpleMC in editable mode:

```powershell
pip install -e ".[dev]"
```

## Quick Start

Define a model that receives a random number generator and returns one simulation outcome.

```python
from simplemc import simulate


def model(rng):
    return rng.normal(10, 2)


result = simulate(
    model,
    n=100_000,
    seed=42,
)

print(result.mean)
print(result.std)
```

Example output:

```text
10.001...
2.001...
```

The same seed produces the same simulation results, making experiments reproducible.

## Monte Carlo Example

SimpleMC can be used to estimate the value of π.

Generate random points inside a square and determine whether each point falls inside the unit circle:

```python
from simplemc import simulate


def inside_circle(rng):
    x = rng.uniform(-1, 1)
    y = rng.uniform(-1, 1)

    return x**2 + y**2 <= 1


result = simulate(
    inside_circle,
    n=100_000,
    seed=42,
)

pi_estimate = 4 * result.mean

print(f"Estimated pi: {pi_estimate:.6f}")
```

Example output:

```text
Estimated pi: 3.142...
```

Increasing the number of simulations generally gives a more precise estimate.

## Simulation Results

`simulate()` returns a `SimulationResult` object.

```python
result = simulate(
    lambda rng: rng.normal(10, 2),
    n=10_000,
    seed=42,
)
```

The result provides basic statistics:

```python
result.values
result.n
result.mean
result.std
result.min
result.max
```

Quantiles can be calculated with:

```python
result.quantile(0.95)
```

For example:

```python
print(f"Mean: {result.mean:.2f}")
print(f"Std:  {result.std:.2f}")
print(f"P95:  {result.quantile(0.95):.2f}")
```

## Reproducibility

SimpleMC uses NumPy's random number generator.

Providing a seed makes the simulation reproducible:

```python
result_1 = simulate(
    model,
    n=10_000,
    seed=42,
)

result_2 = simulate(
    model,
    n=10_000,
    seed=42,
)
```

Both simulations will produce the same values.

Changing the seed produces a different random sample:

```python
result = simulate(
    model,
    n=10_000,
    seed=123,
)
```

## Project Structure

```text
SimpleMC/
│
├── simplemc/
│   ├── __init__.py
│   ├── simulation.py
│   └── result.py
│
├── tests/
│   ├── test_simulation.py
│   └── test_result.py
│
├── examples/
│   └── pi.py
│
├── pyproject.toml
├── README.md
└── .gitignore
```

## Testing

SimpleMC uses `pytest`.

Run the test suite with:

```powershell
pytest
```

The tests cover:

* simulation execution
* result creation
* reproducibility
* random seeds
* input validation
* result statistics
* quantiles

## Current Scope

SimpleMC `0.1.0` contains:

* Monte Carlo simulation
* configurable number of simulations
* reproducible random number generation
* `SimulationResult`
* mean
* standard deviation
* minimum and maximum
* quantiles
* basic input validation

More advanced functionality will be introduced in later releases.

## Version

Current version:

```text
0.1.0
```

## License

SimpleMC is released under the MIT License.
