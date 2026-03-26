"""Solution 09: Use `lax.cond` for branchy tensor logic."""
import jax
import jax.numpy as jnp

from jax import lax

def run_exercise(x):
    return lax.cond(
        jnp.mean(x) > 0,
        lambda t: t * 2,
        lambda t: -t,
        x,
    )


if __name__ == "__main__":
    print(run_exercise)
