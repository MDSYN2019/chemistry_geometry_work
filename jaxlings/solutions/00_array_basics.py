"""Solution 00: Create arrays, reshape, and stack with jax.numpy."""
import jax
import jax.numpy as jnp


def run_exercise():
    a = jnp.arange(6).reshape(2, 3)
    b = jnp.ones((2, 3))
    c = jnp.stack([a, b], axis=0)
    return a, b, c


if __name__ == "__main__":
    print(run_exercise)
