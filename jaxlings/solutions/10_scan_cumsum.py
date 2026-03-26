"""Solution 10: Build cumulative sums with `lax.scan`."""
import jax
import jax.numpy as jnp

from jax import lax

def run_exercise(x):
    def step(carry, elem):
        new_carry = carry + elem
        return new_carry, new_carry
    _, ys = lax.scan(step, 0.0, x)
    return ys


if __name__ == "__main__":
    print(run_exercise)
