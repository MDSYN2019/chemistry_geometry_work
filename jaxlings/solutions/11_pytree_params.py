"""Solution 11: Map over nested parameter pytrees."""
import jax
import jax.numpy as jnp


def run_exercise(params):
    return jax.tree_util.tree_map(lambda v: v * 0.5, params)


if __name__ == "__main__":
    print(run_exercise)
