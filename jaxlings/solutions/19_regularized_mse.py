"""Solution 19: Combine MSE and L2 regularization."""
import jax
import jax.numpy as jnp


def run_exercise(pred, target, params, lam=1e-3):
    mse = jnp.mean((pred - target) ** 2)
    l2 = sum(jnp.sum(v * v) for v in jax.tree_util.tree_leaves(params))
    return mse + lam * l2


if __name__ == "__main__":
    print(run_exercise)
