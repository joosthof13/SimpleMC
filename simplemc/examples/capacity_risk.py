from simplemc import simulate


def capacity_exceeded(rng):
    demand = rng.normal(
        loc=100,
        scale=15,
    )

    capacity = 120

    return demand > capacity


result = simulate(
    capacity_exceeded,
    n=100_000,
    seed=42,
)

probability = result.mean

print(f"Simulations:       {result.n:,}")
print(f"Capacity:          120")
print(f"Probability:       {probability:.2%}")

