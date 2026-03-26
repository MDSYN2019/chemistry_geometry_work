"""Solution 21: Compute Jacobian of a vector function."""
import jax
import jax.numpy as jnp


def run_exercise(x, w):
    f = lambda t: w @ t
    return jax.jacfwd(f)(x)


if __name__ == "__main__":
    print(run_exercise)
