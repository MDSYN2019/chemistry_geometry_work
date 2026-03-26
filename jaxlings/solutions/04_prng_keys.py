"""Solution 04: Split PRNG keys and sample reproducibly."""
import jax
import jax.numpy as jnp


def run_exercise(seed: int = 0):
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)
    a = jax.random.normal(k1, (3,))
    b = jax.random.uniform(k2, (3,))
    return a, b


if __name__ == "__main__":
    print(run_exercise)
