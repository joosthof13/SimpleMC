import numpy as np

from simplemc import simulate


# ---------------------------------------------------------
# 1. Normal simulation
# ---------------------------------------------------------

def normal_model(rng: np.random.Generator) -> float:
    """Generate one normally distributed observation."""
    return rng.normal(loc=10, scale=2)


result = simulate(
    normal_model,
    n=10_000,
    seed=42,
)

print("Normal simulation")
print("------------------")
print(f"n:       {result.n}")
print(f"mean:    {result.mean:.4f}")
print(f"std:     {result.std:.4f}")
print(f"min:     {result.min:.4f}")
print(f"max:     {result.max:.4f}")
print(f"q25:     {result.quantile(0.25):.4f}")
print(f"median:  {result.quantile(0.50):.4f}")
print(f"q75:     {result.quantile(0.75):.4f}")
print()


# ---------------------------------------------------------
# 2. Reproducibility
# ---------------------------------------------------------

result_a = simulate(
    normal_model,
    n=1_000,
    seed=123,
)

result_b = simulate(
    normal_model,
    n=1_000,
    seed=123,
)

print("Reproducibility")
print("----------------")
print("Same seed:", np.array_equal(
    result_a.values,
    result_b.values,
))
print()


# ---------------------------------------------------------
# 3. Different seeds
# ---------------------------------------------------------

result_c = simulate(
    normal_model,
    n=1_000,
    seed=456,
)

print("Different seeds")
print("----------------")
print("Same results:", np.array_equal(
    result_a.values,
    result_c.values,
))
print()


# ---------------------------------------------------------
# 4. Quantiles
# ---------------------------------------------------------

print("Quantiles")
print("---------")

for q in [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]:
    print(f"{q:>4.2f}: {result.quantile(q):.4f}")

print()


# ---------------------------------------------------------
# 5. Error handling: invalid n
# ---------------------------------------------------------

print("Validation tests")
print("-----------------")

try:
    simulate(
        normal_model,
        n=0,
        seed=42,
    )
except ValueError as error:
    print("Invalid n:")
    print(f"  {error}")

print()


# ---------------------------------------------------------
# 6. Error handling: invalid seed
# ---------------------------------------------------------

try:
    simulate(
        normal_model,
        n=100,
        seed="42",
    )
except TypeError as error:
    print("Invalid seed:")
    print(f"  {error}")

print()


# ---------------------------------------------------------
# 7. Error handling: non-numeric model output
# ---------------------------------------------------------

def invalid_model(rng: np.random.Generator):
    return "hello"


try:
    simulate(
        invalid_model,
        n=100,
        seed=42,
    )
except TypeError as error:
    print("Invalid model output:")
    print(f"  {error}")

print()


# ---------------------------------------------------------
# 8. Error handling: NaN
# ---------------------------------------------------------

def nan_model(rng: np.random.Generator):
    return np.nan


try:
    simulate(
        nan_model,
        n=100,
        seed=42,
    )
except ValueError as error:
    print("Non-finite model output:")
    print(f"  {error}")

print()


# ---------------------------------------------------------
# 9. Error handling: model returns multiple values
# ---------------------------------------------------------

def vector_model(rng: np.random.Generator):
    return rng.normal(size=3)


try:
    simulate(
        vector_model,
        n=100,
        seed=42,
    )
except ValueError as error:
    print("Vector model output:")
    print(f"  {error}")