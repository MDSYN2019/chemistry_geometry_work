"""Solution 05: Vectorize a scalar function with `jax.vmap`."""
import jax
import jax.numpy as jnp


def run_exercise():
    f = lambda t: t * t + 1.0
    batched = jax.vmap(f)
    return batched(jnp.array([1., 2., 3.]))


if __name__ == "__main__":
    print(run_exercise)
