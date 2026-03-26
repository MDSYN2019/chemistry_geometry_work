"""Solution 20: Clip gradients to a max global norm."""
import jax
import jax.numpy as jnp


def run_exercise(grads, max_norm=1.0):
    leaves = jax.tree_util.tree_leaves(grads)
    global_norm = jnp.sqrt(sum(jnp.sum(g * g) for g in leaves))
    scale = jnp.minimum(1.0, max_norm / (global_norm + 1e-6))
    return jax.tree_util.tree_map(lambda g: g * scale, grads), global_norm


if __name__ == "__main__":
    print(run_exercise)
