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
print(f"Actual pi:    {3.141592653589793:.6f}")
print(f"Error:        {abs(pi_estimate - 3.141592653589793):.6f}")