"""Solution 01: Practice immutable updates with `.at[...]`."""
import jax
import jax.numpy as jnp


def run_exercise():
    x = jnp.array([0., 1., 2., 3.])
    y = x.at[1:3].set(jnp.array([10., 20.]))
    z = y.at[0].add(5.)
    return x, y, z


if __name__ == "__main__":
    print(run_exercise)
