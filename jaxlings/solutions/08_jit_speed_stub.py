"""Solution 08: Wrap pure math with `jax.jit`."""
import jax
import jax.numpy as jnp


def run_exercise(x):
    @jax.jit
    def f(t):
        return jnp.sin(t) + t * t
    return f(x)


if __name__ == "__main__":
    print(run_exercise)
