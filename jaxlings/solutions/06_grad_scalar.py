"""Solution 06: Differentiate a scalar-valued function with `jax.grad`."""
import jax
import jax.numpy as jnp


def run_exercise(x=2.0):
    f = lambda t: t**3 + 2*t
    g = jax.grad(f)
    return g(x)


if __name__ == "__main__":
    print(run_exercise)
